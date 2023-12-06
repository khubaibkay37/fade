# -*- coding: UTF-8 -*-

import torch
import logging
import os
import numpy as np
import copy
import pickle
import time
from tqdm import tqdm
from random import randint, choice
import torch.nn.functional as F
from torch.utils.data import Dataset as BaseDataset
from torch.nn.utils.rnn import pad_sequence
from typing import NoReturn, List

from utils import utils
from helpers.MetaReader import MetaReader
DEFAULT_EPS = 1e-10

class MetaModel(torch.nn.Module):
    reader = 'MetaReader'
    runner = 'MetaRunner'
    extra_log_args = []

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--model_path', type=str, default='',
                            help='Model save path.')
        parser.add_argument('--num_neg', type=int, default=4,
                            help='The number of negative items for training.')
        parser.add_argument('--num_neg_fair', type=int, default=4,
                            help='The number of negative items for the fairness loss')
        parser.add_argument('--dropout', type=float, default=0,
                            help='Dropout probability for each deep layer')
        parser.add_argument('--buffer', type=int, default=1,
                            help='Whether to buffer feed dicts for dev/test')
        parser.add_argument('--DRM', type=str, default='',
                            help='Use DRM regularization or not.')
        parser.add_argument('--DRM_weight', type=float, default=1,
                            help='DRM term weight.')
        parser.add_argument('--tau', type=float, default=3.0,
                            help='DRM hyperparameter tau.')
        parser.add_argument('--dflag', type=str, default='none',
                            help='Loss weight techniques in terms of dynamic updates.')
        parser.add_argument('--cay', type=float, default=1e-3,
                            help='Decay factor')
        parser.add_argument('--thres', type=float, default=0,
                            help='Correction factor')
        return parser

    @staticmethod
    def init_weights(m):
        if 'Linear' in str(type(m)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif 'Embedding' in str(type(m)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def __init__(self, args, corpus: MetaReader):
        super(MetaModel, self).__init__()
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_path = args.model_path
        self.num_neg = args.num_neg
        self.num_neg_fair = args.num_neg_fair
        self.dropout = args.dropout
        self.buffer = args.buffer
        # self.g = copy.deepcopy(corpus.g)
        self.item_num = corpus.n_items
        self.optimizer = None
        self.check_list = list()  # observe tensors in check_list every check_epoch

        self._define_params()
        self.total_parameters = self.count_variables()
        logging.info('#params: %d' % self.total_parameters)

        self.DRM = args.DRM
        self.DRM_weight = args.DRM_weight
        self.tau = args.tau
        self.dflag = args.dflag
        self.thres = args.thres


    """
    Methods must to override
    """
    # def _define_params(self) -> NoReturn:
    #     self.item_bias = torch.nn.Embedding(self.item_num, 1)

    # def forward(self, feed_dict: dict) -> torch.Tensor:
    #     """
    #     :param feed_dict: batch prepared in Dataset
    #     :return: prediction with shape [batch_size, n_candidates]
    #     """
    #     i_ids = feed_dict['item_id']
    #     prediction = self.item_bias(i_ids)
    #     return prediction.view(feed_dict['batch_size'], -1)


    def get_relevances(self, model, user, items):
        pred_eval = model.model_(user, items, self.DRM)

        return pred_eval.cpu().data.numpy()


    def warp_loss(pos, neg, margin=0.0):
        # neg_highest, _ = neg.max(-1)
        # impostors = torch.log(1.0 + torch.clamp(-pos.unsqueeze(2) + neg.unsqueeze(1) + margin, 0).mean(-1) * n_items).detach()

        # loss_per_pair = torch.clamp(-pos + neg_highest.unsqueeze(-1) + margin, 0)

        loss_per_pair = torch.clamp(-pos + neg + margin, 0)

        #return (impostors * loss_per_pair).sum()
        return loss_per_pair.sum(-1).mean()
    
    def dcg(self, y_pred, y_true, ats=None):
        y_true = y_true.clone()
        y_pred = y_pred.clone()
        actual_length = y_true.shape[1]
        if ats is None:
            ats = [actual_length]
        ats = [min(at, actual_length) for at in ats]

        _, indices = torch.sort(y_pred, descending=True, dim=1)
        y_true = torch.gather(y_true, dim=1, index=indices)

        discounts = (torch.tensor(1.) / torch.log2(torch.arange(actual_length, dtype=torch.float32) + 2.)).to(y_true.device)
        
        discounted_gains = (y_true * discounts)[:, :np.max(ats)]
        cum_dcg = torch.cumsum(discounted_gains, dim=1)
        ats_tensor = torch.tensor(ats, dtype=torch.long) - torch.tensor(1)
        dcg = cum_dcg[:, ats_tensor]

        return dcg

    def sinkhorn_scaling(self, mat, tol=1e-6, max_iter=50):

        for _ in range(max_iter):
            mat = mat / mat.sum(dim=1, keepdim=True).clamp(min=DEFAULT_EPS)
            mat = mat / mat.sum(dim=2, keepdim=True).clamp(min=DEFAULT_EPS)

            if torch.max(torch.abs(mat.sum(dim=2) - 1.)) < tol and torch.max(torch.abs(mat.sum(dim=1) - 1.)) < tol:
                break
        return mat

    def detNeuralSort(self, s, tau=1.0, k=1):
        su = s.unsqueeze(-1).float()
        n = s.size()[1]
        one = torch.ones((n, 1), dtype=torch.float32, device=self._device)

        A_s = torch.abs(su - su.permute(0, 2, 1))

        ones = torch.ones(1, k, device=self._device)
        
        # if torch.cuda.is_available():
        #     ones = ones.cuda()

        B = torch.matmul(A_s, torch.matmul(one, ones))

        #B = torch.matmul(A_s, torch.matmul(one, torch.transpose(one, 0, 1)))
        scaling = (n + 1 - 2 * (torch.arange(n, device=self._device) + 1)).float()
        # if torch.cuda.is_available():
        #     scaling = scaling.cuda()

        C = (su * scaling.unsqueeze(0))[:, :, :k]
        P_max = (C - B).permute(0, 2, 1)
        sm = torch.nn.Softmax(-1)
        P_hat = sm(P_max / tau)
        return P_hat


    def loss(self, predictions, current, data, reduction):
        """
        BPR ranking loss with optimization on multiple negative samples
        @{Recurrent neural networks with top-k gains for session-based recommendations}
        :param predictions: [batch_size, -1], the first column for positive, the rest for negative
        :return:
        """
        sen_attr = current['attr']
        if 'occur' in self.dflag:
            u_occur_time = current['u_occur_time']
            i_occur_time = current['i_occur_time']
        if 'degree' in self.dflag:
            u_degree_weight = current['u_degree_weight']
        if 'recall' in self.DRM:
            num_pos_items = current['num_pos_items']

        pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:1+self.num_neg] # 1 pos : self.num_neg neg

        # if 'attempt0' in self.DRM or 'attempt1' in self.DRM:
            #loss = -((pos_pred[:, None] - neg_pred).sigmoid()).sum(dim=1).log()
            #loss = -torch.mean(torch.log(torch.sigmoid(pos_pred[:,None] - neg_pred)))
        loss = -(pos_pred[:,None] - neg_pred).sigmoid().log().mean(dim=1)

        # else:

        #     neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
        #     loss = -((pos_pred[:, None] - neg_pred).sigmoid() * neg_softmax).sum(dim=1).log()
        #     # neg_pred = (neg_pred * neg_softmax).sum(dim=1)
        #     # loss = F.softplus(-(pos_pred - neg_pred)).mean()
        #     # ↑ For numerical stability, we use 'softplus(-x)' instead of '-log_sigmoid(x)'


        if 'test' not in self.dflag or data.phase == 'test':
            if 'main' in self.dflag:
                if 'uoccur' in self.dflag:
                    loss = loss * u_occur_time
                elif 'ioccur' in self.dflag:
                    loss = loss * i_occur_time
                elif 'udegree' in self.dflag:
                    loss = loss * u_degree_weight

        if reduction == 'mean':
            loss = loss.mean()

        #print(u_degree_weight)

        loss_ = 0
        fl = 0
        # if self.DRM == 'acc':
        #     tau = 1
        #     _k = 1 # just one postiive item
        #     p_hat = self.detNeuralSort(predictions, tau=tau, k=_k)
        #     ps = p_hat.sum(1).clamp(0, 1)
        #     a = ps[:, :_k]
        #     b = ps[:, _k:]
        #     loss1 = ((a - 1)**2).sum(-1) + (b ** 2).sum(-1)

        #     if 'test' not in self.dflag or data.phase == 'test':
        #         if 'uoccur' in self.dflag:
        #             loss1 = loss1 * u_occur_time
        #         elif 'ioccur' in self.dflag:
        #             loss1 = loss1 * i_occur_time
        #         elif 'udegree' in self.dflag:
        #             loss1 = loss1 * u_degree_weight

        #     if reduction == 'mean':
        #         loss1 = loss1.mean()

        #     loss = loss + self.DRM_weight * loss1

        if 'none' not in self.DRM:
            _k = 1

            adv = sen_attr == 0 # Male
            disadv = sen_attr != 0 # Female

            fairness_loss = []

            for bool_mask in [adv, disadv]:
                new_predictions = predictions[bool_mask]
                #new_num_pos_items = num_pos_items[bool_mask]

                if 'REC' in self.DRM:
                    ps = new_predictions
                    a = ps[:, :_k]
                    loss1 = a.sum(-1)

                elif 'neural' in self.DRM:
                    p_hat = self.detNeuralSort(new_predictions, tau=self.tau, k=new_predictions.size()[1])

                    #print('p_hat', p_hat.shape)

                    # If there are only M/F users in the mini-batch
                    if p_hat.size()[0] == 0:
                        #print('there are only M/F users in the mini-batch!!!')
                        return loss, loss, None, None, None
                    # if np.isnan(loss1.cpu().data.numpy()).any():
                    #     fairness_loss = [0,0]
                    #     fl = torch.zeros(1).to(self._device)
                    #     flag = 1
                    #     #print('there are only M/F users in the mini-batch!!!')

                    #     return loss, loss, None, None

                    p_hat = self.sinkhorn_scaling(p_hat)

                    y_true = torch.zeros(p_hat.size()[0], p_hat.size()[1]).to(self._device)
                    y_true[:,:_k] = 1

                    ground_truth = torch.matmul(p_hat, y_true.unsqueeze(-1)).squeeze(-1)
                    discounts = (torch.tensor(1) / torch.log2(torch.arange(ground_truth.shape[1], dtype=torch.float) + 2.0)).to(
                        device=ground_truth.device)
                    discounted_gains = ground_truth * discounts
                    
                    idcg = self.dcg(y_true, y_true).permute(1,0)
                    ndcg = discounted_gains.sum(dim=-1) / (idcg + DEFAULT_EPS)
                    loss1 = ndcg

                elif 'approx' in self.DRM:
                    #alpha = 10
                    alpha = self.tau # use tau for alpha to avoid additional implementaion of hyperparameter
                    # {10, 20, 50, 100, 150, 200, 250, 300}

                    y_pred = new_predictions
                    if y_pred.size()[0] == 0:
                        return loss, loss, None, None, None
                    
                    y_true = torch.zeros(y_pred.size()[0], y_pred.size()[1]).to(self._device)
                    y_true[:,:_k] = 1
                    y_true_sorted = y_true # already sorted
                    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
                    true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)

                    true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
                    padded_pairs_mask = torch.isfinite(true_diffs)
                    padded_pairs_mask.diagonal(dim1=-2, dim2=-1).zero_()

                    pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(y_pred.device)
                    D = torch.log2(1. + pos_idxs.float())[None, :]
                    maxDCGs = torch.sum(y_true_sorted / D, dim=-1).clamp(min=DEFAULT_EPS)
                    G = true_sorted_by_preds / maxDCGs[:, None]

                    scores_diffs = (y_pred_sorted[:,:,None] - y_pred_sorted[:,None,:])
                    # scores_diffs[~padded_pairs_mask] = 0.
                    approx_pos = 1. + torch.sum(padded_pairs_mask.float() * (torch.sigmoid(-alpha * scores_diffs).clamp(min=DEFAULT_EPS)), dim=-1)
                    approx_D = torch.log2(1. + approx_pos)
                    # why G... why not true_sorted_by_preds according to the paper?
                    approx_NDCG = torch.sum((G / approx_D), dim=-1)
                    loss1 = approx_NDCG

                elif 'listnet' in self.DRM: 
                    y_pred = new_predictions
                    if y_pred.size()[0] == 0:
                        return loss, loss, None, None, None
                    
                    y_true = torch.zeros(y_pred.size()[0], y_pred.size()[1]).to(self._device)
                    y_true[:,:_k] = 1

                    preds_smax = F.softmax(y_pred, dim=1)
                    true_smax = F.softmax(y_true, dim=1)
                    preds_smax = preds_smax + DEFAULT_EPS

                    if 'typea' in self.DRM:
                        preds_log = torch.log(preds_smax)
                    elif 'typeb' in self.DRM:
                        preds_log = preds_smax
                    loss1 = torch.sum(true_smax * preds_log, dim=1)


                else:
                    if new_predictions.size()[0] == 0:
                        return loss, loss, None, None, None
                    p_hat = self.detNeuralSort(new_predictions, tau=self.tau, k=_k)
                    ps = p_hat.sum(1).clamp(0, 1)
                    a = ps[:, :_k]
                    b = ps[:, _k:_k+self.num_neg_fair]

                    # self.DRM != 100:
                    if 'sqr' in self.DRM:
                        loss1 = (a**2).sum(-1) + ((b-1)**2).sum(-1)
                    elif 'sim' in self.DRM:
                        loss1 = a.sum(-1) + (1-b).sum(-1)
                    elif 'bpr' in self.DRM:
                        loss1 = (a-b).sigmoid().sum(dim=1).log()
                    elif 'onlypos' in self.DRM:
                        loss1 = a.sum(-1)

                    # neg_softmax = (b - b.max()).softmax(dim=1)
                    # loss1 = -((a[:, None] - b).sigmoid() * neg_softmax).sum(dim=1).log()#.mean()

                # if 'test' not in self.dflag or data.phase == 'test':
                #     if 'fair' in self.dflag:
                #         if 'uoccur' in self.dflag:
                #             g_weights = u_occur_time[bool_mask]
                #         elif 'ioccur' in self.dflag:
                #             g_weights = i_occur_time[bool_mask]
                #         elif 'udegree' in self.dflag:
                #             g_weights = u_degree_weight[bool_mask]

                #         loss1 = loss1 * g_weights

                if 'recall' in self.DRM:
                    new_num_pos_items = num_pos_items[bool_mask]
                    loss1 = loss1 / new_num_pos_items

                if reduction == 'mean':
                    loss1 = loss1.mean()


                fairness_loss.append(loss1)

                # If there are only M/F users in the mini-batch
                if np.isnan(loss1.cpu().data.numpy()).any():
                    fairness_loss = [0,0]
                    fl = torch.zeros(1).to(self._device)
                    flag = 1
                    #print('there are only M/F users in the mini-batch!!!')

                    return loss, loss, None, None, None

            # Types of loss functions
            if 'diff' in self.DRM:
                fl = (fairness_loss[1] - fairness_loss[0])**2
                fl = -((fl).sigmoid()).log()
            elif 'log' in self.DRM:
                diff = fairness_loss[0] - fairness_loss[1]
                fl = -((-diff).sigmoid()).log()
            elif 'opposite' in self.DRM:
                diff = fairness_loss[0] - fairness_loss[1]
                fl = -((diff).sigmoid()).log()
            elif 'allnegative' in self.DRM:
                if fairness_loss[1] - fairness_loss[0] <= 0:
                    diff = fairness_loss[1] - fairness_loss[0]
                else:
                    diff = fairness_loss[0] - fairness_loss[1]
                fl = -((diff).sigmoid()).log()
            elif 'absolute' in self.DRM:
                diff = fairness_loss[0] - fairness_loss[1]
                #pd = abs(diff+self.correction)
                pd = abs(diff)
                fl = -((-pd).sigmoid()).log()
            elif 'zerostop' in self.DRM:
                if fairness_loss[1] - fairness_loss[0] <= 0:
                    diff = fairness_loss[1] - fairness_loss[0]
                    fl = -((diff).sigmoid()).log()
                else:
                    fl = 0
            elif 'clamp' in self.DRM:
                fl = -((torch.clamp(fairness_loss[1] - fairness_loss[0], max=0)).sigmoid()).log()
            elif 'hinge' in self.DRM:
                if 'zero' in self.DRM:
                    margin = 0.0
                elif 'one' in self.DRM:
                    margin = 1.0
                fl = -((torch.clamp(-fairness_loss[1] + fairness_loss[0] + margin, 0)).sigmoid()).log()
            else:
                print('@@@@@@@@@loss error@@@@@@@@@@')

            loss_ = loss


            lambda_ = self.DRM_weight

            # all parameters
            if self.DRM_weight == -2:
                # calculate gradients of 'loss' w.r.t. model parameters
                gradient_loss_wrt_param = torch.autograd.grad(loss, self.parameters(), retain_graph=True)[0]
                gradient_diff_wrt_param = torch.autograd.grad(diff, self.parameters(), retain_graph=True)[0]
                # flatten the gradients
                gradient_loss_wrt_param = torch.flatten(gradient_loss_wrt_param)
                gradient_diff_wrt_param = torch.flatten(gradient_diff_wrt_param)
                #print('gradient loss shape', gradient_loss_wrt_param.shape)
                #print('gradient diff shape', gradient_diff_wrt_param.shape)

                lambda_ = (-1) * torch.sum(gradient_loss_wrt_param*gradient_diff_wrt_param)/torch.sum(gradient_diff_wrt_param*gradient_diff_wrt_param)
                #print(lambda_)
                # multiply 1 / (sigmoid of thres)
                lambda_ *= (1 + np.exp(-self.thres))


                # make the zero value to torch
                if lambda_ < 0:
                    lambda_ = torch.tensor(0).to(self._device)
            
            # selected parameters
            elif self.DRM_weight == -3:
                gradient_loss = torch.autograd.grad(loss, predictions, retain_graph=True)[0]
                gradient_diff = torch.autograd.grad(diff, predictions, retain_graph=True)[0]
                gradient_loss = torch.flatten(gradient_loss)
                gradient_diff = torch.flatten(gradient_diff)
                lambda_ = -2 * torch.sum(gradient_loss*gradient_diff)/torch.sum(gradient_diff*gradient_diff)
                #print(lambda_)
            
            # elif self.DRM_weight == -1:
            #     # calculate gradients of "loss" and "diff" and 
            #     gradient_loss = torch.autograd.grad(loss, predictions, retain_graph=True)[0]
            #     gradient_diff = torch.autograd.grad(diff, predictions, retain_graph=True)[0]
            #     #print('gradient_loss shape', gradient_loss.shape)
            #     #print('gradient_diff shape', gradient_diff.shape)

            #     # calculate dot product between them
            #     gradient_dot = torch.sum(gradient_loss * gradient_diff, dim=1)
            #     #print('gradient_dot shape', gradient_dot.shape)
            #     # calculate L2 norm of gradient_diff
            #     # gradient_diff_norm = torch.norm(gradient_diff, p=2, dim=1)
            #     # print('gradient_diff_norm shape', gradient_diff_norm.shape)
            #     # calculate
            #     lambda_ = -2*gradient_dot/torch.sum(gradient_diff * gradient_diff, dim=1)
            #     lambda_ = lambda_.mean()
                #print('lambda_ shape', lambda_.shape)
                #print('lambda: ', lambda_)

            


            if 'justwatch' not in self.DRM:
                loss = loss + lambda_ * fl

            if fl == 0:
                fl_return = None
            else:
                fl_return = fl * lambda_

            if self.DRM_weight < 0:
                lamb = lambda_
            else:
                lamb = None
            #lambda_ if self.DRM_weight < 0 else None
            return loss, loss_, fl_return, diff, lamb

        return loss, loss, None, None, None



    def customize_parameters(self) -> list:
        # customize optimizer settings for different parameters
        weight_p, bias_p = [], []
        for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()):
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        optimize_dict = [{'params': weight_p}, {'params': bias_p, 'weight_decay': 0}]
        return optimize_dict

    """
    Auxiliary methods
    """
    def save_model(self, model_path=None, add_path=None) -> NoReturn:
        if model_path is None:
            model_path = self.model_path
        if add_path:
            model_path += add_path
        utils.check_dir(model_path)
        torch.save({'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                    model_path)
        #logging.info('Save model to ... ' + model_path[50:])


    def save_best_model(self, model_path=None) -> NoReturn:
        if model_path is None:
            model_path = self.model_path
        utils.check_dir(model_path)
        torch.save({'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                    model_path + '_best')
        #logging.info('Save model to ' + model_path[:50] + '...')

    def load_model(self, model_path=None, add_path=None, flag=0) -> NoReturn:
        if model_path is None:
            model_path = self.model_path
        if add_path:
            model_path += add_path
        
        if torch.cuda.is_available():
            check_point = torch.load(model_path)
        else:
            check_point = torch.load(model_path, map_location=torch.device('cpu'))
            
        self.load_state_dict(check_point['model_state_dict'])
        if flag == 0:
            self.optimizer.load_state_dict(check_point['optimizer_state_dict'])
        #logging.info('Load model from ' + model_path)

    def count_variables(self) -> int:
        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_parameters

    def actions_before_train(self):  # e.g. re-initial some special parameters
        pass

    def actions_after_train(self):  # e.g. save selected parameters
        pass

    """
    Define dataset class for the model
    """

    class Dataset(BaseDataset):
        def __init__(self, model, args, corpus, phase, add_n_bat=0):
            self.model = model  # model object reference
            self.corpus = corpus  # reader object reference
            self.phase = phase
            self.neg_items = None # if phase == 'train' else self.data['neg_items']
            # ↑ Sample negative items before each epoch during training
            self.train_ratio = args.train_ratio
            self.mini_batch_path = corpus.mini_batch_path
            # self.graph_path = corpus.graph_path
            self.batch_size = args.batch_size
            self.buffer = self.model.buffer and self.phase != 'train'

            self.train_boundary = corpus.n_train_batches
            self.snapshots_path = corpus.snapshots_path
            self.n_snapshots = corpus.n_snapshots
            
            if phase == 'fulltrain':
                self.train_boundary += add_n_bat
                self.n_batches = self.train_boundary
                logging.info("full-train n_batches: %s" %str(self.n_batches))
            elif phase == 'train':
                self.n_batches = self.train_boundary
                logging.info("fine-tuning: (pre)train n_batches: %s" %str(self.n_batches))
            elif phase == 'test':
                self.n_batches = corpus.n_batches-self.train_boundary
                #self._prepare_neg_items()
                logging.info("fine-tuning: test n_batches: %s" %str(self.n_batches))
                #assert corpus.n_test == len(self.neg_items), "Neg items not equal"


            user_attr = utils.read_data_from_file_int(corpus.user_attr_path)
            self.user_attr_dict = {}
            for user in user_attr:
                self.user_attr_dict[user[0]] = user[1] # gender M: 1, F: 0

            #self._set_user_item_occurrence_time()
            #self._set_user_item_historical_degree()
            #self.decay_factor = 1e-3
            self.decay_factor = args.cay
            self.dflag = args.dflag
            self.DRM = args.DRM

            #decay_mat = np.exp(self.decay_factor * (his_time_mat - cur_time))

            self.fine_tune_snap_idx = 0
            self._set_user_intersactions_at_each_snapshot()


        def _set_user_intersactions_at_each_snapshot(self):
            # if self.fine_tune_snap_idx == 0:
            #     interaction_file = os.path.join(self.snapshots_path, 'remain_train_snap{}'.format(self.fine_tune_snap_idx))
            # # fine-tuning data 1~9
            # else:
            #     interaction_file = os.path.join(self.snapshots_path, 'next_test_snap{}'.format(self.fine_tune_snap_idx))

            self.num_interactions_snapshots = {}
            for i in range(self.n_snapshots):
                if i == 0:
                    pass
                    #interaction_file = os.path.join(self.snapshots_path, 'remain_train_snap{}'.format(i))
                else:
                    interaction_file = os.path.join(self.snapshots_path, 'next_test_snap{}'.format(i-1))
                    train_edges = utils.read_data_from_file_int(interaction_file)
                    self.num_interactions_snapshots[i] = utils.get_user_dil_from_edgelist(train_edges)



        def _set_user_item_occurrence_time(self):
            self.u_occur_time = {}
            self.i_occur_time = {}

            df = self.corpus.data_df.loc[:self.train_boundary*self.batch_size,:].values.astype(np.int64)
            time_period = 0

            n_interaction_per_snapshot = self.corpus.n_batches_per_snapshot*self.batch_size
            #print('n_batches_per_snapshot: {}, n_interaction_per_snapshot: {}'.format(self.corpus.n_batches_per_snapshot, n_interaction_per_snapshot))
            for batch_cnt, interaction in enumerate(df):
                if self.u_occur_time.get(interaction[0]) is None:
                    self.u_occur_time[interaction[0]] = time_period

                if self.i_occur_time.get(interaction[1]) is None:
                    self.i_occur_time[interaction[1]] = time_period

                if batch_cnt % n_interaction_per_snapshot == n_interaction_per_snapshot-1:
                    time_period += 1

                #print(batch_cnt)

            # Save the latest time_period in the dict by a key of "-1"
            self.u_occur_time[-1] = time_period+1
            self.i_occur_time[-1] = time_period+1
            self.u_buffer = []
            self.i_buffer = []

            #print('start of the test time period: {}'.format(time_period+1))

        # def _set_normalized_occurence_time(self):
        #     self.normalized_u_occur_time = {}

        def update_occurrence_time(self):

            new_users = set(self.u_buffer)
            for user in new_users:
                self.u_occur_time[user] = self.u_occur_time[-1]
            self.u_occur_time[-1] += 1
            self.u_buffer = []

            new_items = set(self.i_buffer)
            for item in new_items:
                self.i_occur_time[item] = self.i_occur_time[-1]
            self.i_occur_time[-1] += 1
            self.i_buffer = []

            #print('update the test time period: {}'.format(self.u_occur_time[-1]))

        def _set_user_item_historical_degree(self):
            self.u_degree = {}  
            for user, items in self.corpus.user_hist_set.items():
                self.u_degree[user] = len(items)

            self.i_degree = {}
            for item, users in self.corpus.item_hist_set.items():
                self.i_degree[item] = len(users)

            #print('u_degree: {}'.format(self.u_degree))

            #self.new_interaction_buffer = []

        def update_historical_degree(self, users, items):
            for u in users:
                if self.u_degree.get(u.item()) is None:
                    self.u_degree[u.item()] = 0
                self.u_degree[u.item()] += 1

            for i in items:
                if self.i_degree.get(i.item()) is None:
                    self.i_degree[i.item()] = 0
                self.i_degree[i.item()] += 1

        def __len__(self):
            '''
            Returns the number of batches
            '''
            if self.phase == 'train':
                return self.n_batches
            else:
                return self.n_batches


        def __getitem__(self, index: int) -> dict:
            #last = self._get_feed_dict(index)
            current = self._get_feed_dict(index, current=True)
            # current['idx'] = index

            return current

        def _get_feed_dict(self, index: int, current=False) -> dict:
            r"""Return user-item mini-batch and index/value adjacency-batch (torch.FloatTensor).

            Process:
                train
                    last [batch_size*2, 2]
                        randomly sample positive/negative user/item among history
                        requires user_hist_set, item_hist_set
                        use current batch to index user/item

                    current [batch_size, 2]
                        positive: read from saved mini-batch
                        negative: randomly pick using self._sample_items

                test
                    index + n_batches*train_ratio
                    last [batch_size*2, 2]
                        let's prepare everything needed here
                        positive: dictionary based. if not, use current add user
                        negative: randomly sample negative item
                        use current batch to index user/item
                        requires preliminary dictionary

                    current [batch_size, 1+99=100 ?]
                        positive: read from saved mini-batch
                        negative: already prepared negative item using self._prepare_neg_items

            Input:
                index: index of the batch

            """


            if self.phase == 'test':
                index += self.train_boundary

            if current:  # return [batch_size, -1]
                user_id, item_id = torch.load(os.path.join(self.mini_batch_path, str(index)+'.pt')).T
                neg_items = self._sample_neg_items(index*self.batch_size,
                                                  index*self.batch_size+len(user_id))
                item_id_ = torch.cat((item_id.reshape(-1, 1), neg_items), axis=-1)
                feed_dict = {'user_id': user_id, #(batch_size, )
                             'item_id': item_id_} #(batch_size, 1+neg_items)
                
                # print('user_id: {}'.format(user_id.shape))
                # print('item_id: {}'.format(item_id_.shape))

                sen_attr = []
                for user in user_id:
                    sen_attr.append(self.user_attr_dict[user.item()])
                sen_attr = torch.from_numpy(np.array(sen_attr))
                feed_dict['attr'] = sen_attr

                # the number of positive items of each user
                # feed_dict['u_degree'] = torch.from_numpy(np.array([self.u_degree[u.item()] for u in user_id]))

                #print('fine_tune_snap_idx: {}'.format(self.fine_tune_snap_idx))
                if 'recall' in self.DRM:
                    pos_items = []
                    if self.fine_tune_snap_idx == 0:
                        for user in user_id:
                            pos_items.append(len(self.corpus.user_hist_set[user.item()]))
                    else:
                        for user in user_id:
                            if self.num_interactions_snapshots[self.fine_tune_snap_idx].get(user.item()) is None:
                                #print(user.item())
                                pos_items.append(1)
                            else:
                                pos_items.append(len(self.num_interactions_snapshots[self.fine_tune_snap_idx][user.item()]))
                    pos_items = torch.from_numpy(np.array(pos_items))
                    feed_dict['num_pos_items'] = pos_items


                # exp attenuation 
                # User occurrence time
                if 'occur' in self.dflag:
                    cur_time = self.u_occur_time[-1]
                    user_occur_time = []
                    for user in user_id:
                        if self.u_occur_time.get(user.item()) is None:
                            # cold-start user -> final value: 1
                            user_occur_time.append(cur_time)

                            # To update u_occur_time later
                            self.u_buffer.append(user.item())
                        else:
                            user_occur_time.append(self.u_occur_time[user.item()])

                    normalized_u_occur_time = np.exp(self.decay_factor * (np.array(user_occur_time) - np.array(cur_time)))
                    feed_dict['u_occur_time'] = torch.from_numpy(normalized_u_occur_time)

                    # Item occurrence time
                    cur_time = self.i_occur_time[-1]
                    item_occur_time = []
                    for item in item_id:
                        if self.i_occur_time.get(item.item()) is None:
                            # cold-start item -> final value: 1
                            item_occur_time.append(cur_time)

                            # To update u_occur_time later
                            self.i_buffer.append(item.item())
                        else:
                            item_occur_time.append(self.i_occur_time[item.item()])

                    normalized_i_occur_time = np.exp(self.decay_factor * (np.array(item_occur_time) - np.array(cur_time)))
                    feed_dict['i_occur_time'] = torch.from_numpy(normalized_i_occur_time)


                # Degree information; the lower the degree is, the higher the weight is.
                #normalized_u_degree_importance = []
                if 'degree' in self.dflag:
                    #max_degree = self.u_degree.max()
                    user_degree = []
                    for user in user_id:
                        if self.u_degree.get(user.item()) is None:
                            user_degree.append(0)
                        else:
                            user_degree.append(self.u_degree[user.item()])

                    normalized_u_degree_importance = np.exp(self.decay_factor * (np.array(user_degree) * (-1)))
                    feed_dict['u_degree_weight'] = normalized_u_degree_importance


                return feed_dict

            # else: # return [batch_size*2, 2]
            #     if self.phase == 'train': # randomly sample negative items
            #         user_id, item_id = torch.load(os.path.join(self.mini_batch_path, str(index)+'.pt')).T
            #         # Exception handling: If a user has no previous interactions, then the current item is selected

            #         # same user, different item: (u, i'), (u, -i')
            #         # for positive: get user, and pick another used item
            #         # for negative: get user, and pick random negative item
            #         pos_items_u = torch.zeros(size=(len(user_id), 1), dtype=torch.int64)
            #         neg_items_u = torch.zeros(size=(len(user_id), 1), dtype=torch.int64)
            #         for idx, user in enumerate(user_id):
            #             user_hist_set = copy.deepcopy(self.corpus.user_hist_set[user.item()])
            #             user_clicked_set = copy.deepcopy(self.corpus.user_clicked_set[user.item()])
            #             pos_items_u[idx] = choice(user_hist_set)
            #             neg_items_u[idx] = self._randint_w_exclude(user_clicked_set)
            #         items_u = torch.cat((pos_items_u, neg_items_u), axis=-1)

            #         # different user, same item: (u', i), (u', -i)
            #         # for positive: get another user, and pick item
            #         # for negative: get another user, and pick random negative item
            #         user_id_ = torch.zeros_like(user_id)
            #         for idx, item in enumerate(item_id): # pick u'
            #             user_id_[idx] = choice(self.corpus.item_hist_set[item.item()])

            #         neg_items_u_ = torch.zeros(size=(len(user_id), 1), dtype=torch.int64)
            #         for idx, user in enumerate(user_id_):
            #             user_clicked_set = copy.deepcopy(self.corpus.user_clicked_set[user.item()])
            #             neg_items_u_[idx] = self._randint_w_exclude(user_clicked_set)
            #         items_u_ = torch.cat((item_id.reshape(-1, 1), neg_items_u_), axis=-1)

            #         user_id = torch.cat((user_id, user_id_), axis=0)
            #         item_id = torch.cat((items_u, items_u_), axis=0)
            #         feed_dict = {'user_id': user_id, # [batch_size*2,]
            #                      'item_id': item_id} # [batch_size*2, 2]

            #         return feed_dict

            #     else: # read saved positive-negative pairs. Prepared in helpers/MetaReader.py
            #         feed_dict = torch.load(os.path.join(self.corpus.last_batch_path, str(index)+'.pt'))
            #         return feed_dict


        def _sample_neg_items(self, index, index_end):
            #num_neg = self.model.num_neg
            num_neg = max(self.model.num_neg, self.model.num_neg_fair)

            neg_items = torch.zeros(size=(index_end-index, num_neg), dtype=torch.int64)
            for idx, user in enumerate(self.corpus.user_list[index:index_end]): # Automatic coverage?
                user_clicked_set = copy.deepcopy(self.corpus.user_clicked_set[user])
                # By copying, it may not collide with other process with same user index
                for neg in range(num_neg):
                    neg_item = self._randint_w_exclude(user_clicked_set)
                    neg_items[idx][neg] = neg_item
                    # Skip below: one neg for train
                    user_clicked_set = np.append(user_clicked_set, neg_item)

            return neg_items

        def _randint_w_exclude(self, clicked_set):
            randItem = randint(1, self.corpus.n_items-1)
            return self._randint_w_exclude(clicked_set) if randItem in clicked_set else randItem
