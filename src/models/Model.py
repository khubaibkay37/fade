# -*- coding: UTF-8 -*-

import torch
import logging
import os
import numpy as np
import copy
from random import randint
from torch.utils.data import Dataset as BaseDataset
from typing import NoReturn, List

from utils import utils
from helpers.Reader import Reader
DEFAULT_EPS = 1e-10

class Model(torch.nn.Module):
    reader = 'Reader'
    runner = 'Runner'
    extra_log_args = []

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--model_path', type=str, default='',
                            help='Model save path.')
        parser.add_argument('--num_neg', type=int, default=4,
                            help='The number of negative items for training.')
        parser.add_argument('--num_neg_fair', type=int, default=4,
                            help='The number of negative items for the fairness loss')
        parser.add_argument('--DRM', type=str, default='',
                            help='Use DRM regularization or not.')
        parser.add_argument('--DRM_weight', type=float, default=1,
                            help='DRM term weight.')
        parser.add_argument('--tau', type=float, default=3.0,
                            help='DRM hyperparameter tau.')
        return parser

    @staticmethod
    def init_weights(m):
        if 'Linear' in str(type(m)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif 'Embedding' in str(type(m)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def __init__(self, args, corpus: Reader):
        super(Model, self).__init__()
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_path = args.model_path
        self.num_neg = args.num_neg
        self.num_neg_fair = args.num_neg_fair
        self.item_num = corpus.n_items
        self.optimizer = None
        self._define_params()
        self.total_parameters = self.count_variables()
        logging.info('#params: %d' % self.total_parameters)

        self.DRM = args.DRM
        self.DRM_weight = args.DRM_weight
        self.tau = args.tau


    def get_relevances(self, model, user, items):
        pred_eval = model.model_(user, items, self.DRM)

        return pred_eval.cpu().data.numpy()

    def detNeuralSort(self, s, tau=1.0, k=1):
        su = s.unsqueeze(-1).float()
        n = s.size()[1]
        one = torch.ones((n, 1), dtype=torch.float32, device=self._device)
        A_s = torch.abs(su - su.permute(0, 2, 1))
        ones = torch.ones(1, k, device=self._device)
        B = torch.matmul(A_s, torch.matmul(one, ones))
        #B = torch.matmul(A_s, torch.matmul(one, torch.transpose(one, 0, 1)))
        scaling = (n + 1 - 2 * (torch.arange(n, device=self._device) + 1)).float()
        C = (su * scaling.unsqueeze(0))[:, :, :k]
        P_max = (C - B).permute(0, 2, 1)
        sm = torch.nn.Softmax(-1)
        P_hat = sm(P_max / tau)
        return P_hat

    def loss(self, predictions, current, data, reduction):

        sen_attr = current['attr']
        pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:1+self.num_neg] # 1 pos : self.num_neg neg
        loss = -(pos_pred[:,None] - neg_pred).sigmoid().log().mean(dim=1)
        if reduction == 'mean':
            loss = loss.mean()

        loss_ = 0
        fl = 0
        if 'none' not in self.DRM:
            _k = 1
            adv = sen_attr == 0 # Male
            disadv = sen_attr != 0 # Female
            fairness_loss = []

            for bool_mask in [adv, disadv]:
                new_predictions = predictions[bool_mask]

                # If there are only M/F users in the mini-batch
                if new_predictions.size()[0] == 0:
                    return loss, loss, None, None
                p_hat = self.detNeuralSort(new_predictions, tau=self.tau, k=_k)
                ps = p_hat.sum(1).clamp(0, 1)
                a = ps[:, :_k]
                b = ps[:, _k:_k+self.num_neg_fair]
                loss1 = a.sum(-1)
                if reduction == 'mean':
                    loss1 = loss1.mean()
                fairness_loss.append(loss1)

            # Types of loss functions
            if 'log' in self.DRM:
                diff = fairness_loss[0] - fairness_loss[1]
                fl = -((-diff).sigmoid()).log()

            elif 'absolute' in self.DRM:
                diff = fairness_loss[0] - fairness_loss[1]
                #pd = abs(diff+self.correction)
                pd = abs(diff)
                fl = -((-pd).sigmoid()).log()


            loss_ = loss
            #lambda_ = self.DRM_weight
            loss = loss + self.DRM_weight * fl

            if fl == 0:
                fl_return = None
            else:
                fl_return = fl * self.DRM_weight

            return loss, loss_, fl_return, diff

        return loss, loss, None, None

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


    """
    Define dataset class for the model
    """
    class Dataset(BaseDataset):
        def __init__(self, model, args, corpus, phase, add_n_bat=0):
            self.model = model  # model object reference
            self.corpus = corpus  # reader object reference
            self.phase = phase
            self.train_ratio = args.train_ratio
            self.mini_batch_path = corpus.mini_batch_path
            self.batch_size = args.batch_size
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
                logging.info("fine-tuning: test n_batches: %s" %str(self.n_batches))
                #assert corpus.n_test == len(self.neg_items), "Neg items not equal"

            user_attr = utils.read_data_from_file_int(corpus.user_attr_path)
            self.user_attr_dict = {}
            for user in user_attr:
                self.user_attr_dict[user[0]] = user[1] # gender M: 1, F: 0
            self.DRM = args.DRM

        def __len__(self):
            return self.n_batches

        def __getitem__(self, index: int) -> dict:
            current = self._get_feed_dict(index, self.phase)
            return current

        def _get_feed_dict(self, index: int) -> dict:

            if self.phase == 'test':
                index += self.train_boundary

            user_id, item_id = torch.load(os.path.join(self.mini_batch_path, self.phase, str(index)+'.pt')).T
            neg_items = self._sample_neg_items(index*self.batch_size,
                                                index*self.batch_size+len(user_id))
            item_id_ = torch.cat((item_id.reshape(-1, 1), neg_items), axis=-1)
            feed_dict = {'user_id': user_id, #(batch_size, )
                            'item_id': item_id_} #(batch_size, 1+neg_items)

            sen_attr = []
            for user in user_id:
                sen_attr.append(self.user_attr_dict[user.item()])
            sen_attr = torch.from_numpy(np.array(sen_attr))
            feed_dict['attr'] = sen_attr

            return feed_dict

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
