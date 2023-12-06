# coding=utf-8
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
import torch.nn as nn

from utils import utils
from helpers.MetaReader import MetaReader


class AdverModel(nn.Module):
    reader = 'MetaReader'
    runner = 'AdverRunner'
    extra_log_args = []

    @staticmethod
    def parse_model_args(parser, model_name='BaseModel'):
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
        parser.add_argument('--tau', type=float, default=1.0,
                            help='DRM hyperparameter tau.')
        parser.add_argument('--dflag', type=str, default='',
                            help='Loss weight techniques in terms of dynamic updates.')
        parser.add_argument('--cay', type=float, default=1e-3,
                            help='Decay factor')
        return parser

    @staticmethod
    def init_weights(m):
        if 'Linear' in str(type(m)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif 'Embedding' in str(type(m)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def __init__(self, args, corpus):
        super(AdverModel, self).__init__()
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_path = args.model_path
        self.num_neg = args.num_neg
        self.num_neg_fair = args.num_neg_fair
        self.dropout = args.dropout
        #self.buffer = args.buffer
        # self.g = copy.deepcopy(corpus.g)
        #self.item_num = corpus.n_items
        self.optimizer = None
        #self.check_list = list()  # observe tensors in check_list every check_epoch

        self._define_params()
        self.total_parameters = self.count_variables()
        logging.info('#params: %d' % self.total_parameters)

        self.DRM = args.DRM
        self.DRM_weight = args.DRM_weight
        self.tau = args.tau
        self.dflag = args.dflag


        # self.overall_dp = overall_dp
        # self.user_num = user_num
        # self.item_num = item_num
        # self.u_vector_size = u_vector_size
        # self.i_vector_size = i_vector_size
        # self.dropout = dropout
        # self.random_seed = random_seed
        # self.filter_mode = filter_mode
        # torch.manual_seed(self.random_seed)
        # torch.cuda.manual_seed(self.random_seed)
        # self.model_path = model_path

        # self._init_nn()
        self._init_sensitive_filter()
        # logging.debug(list(self.parameters()))

        # self.total_parameters = self.count_variables()
        # logging.info('# of params: %d' % self.total_parameters)

        # # optimizer assigned by *_runner.py
        # self.optimizer = None

    # def _init_nn(self):
    #     """
    #     Initialize neural networks
    #     :return:
    #     """
    #     raise NotImplementedError

    def _init_sensitive_filter(self):
        def get_sensitive_filter(embed_dim):
            sequential = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.LeakyReLU(),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LeakyReLU(),
                nn.BatchNorm1d(embed_dim)
            )
            return sequential
        # num_features = len(self.overall_dp.feature_columns)
        # self.feature_info = self.overall_dp.feature_info
        # #self.filter_num = num_features if self.filter_mode == 'combine' else 2**num_features  # to be modified
        # if self.filter_mode == 'separate':
        #     self.filter_num = 2**num_features
        # else:
        #     self.filter_num = num_features

        self.num_features = 2
        self.sens_filter = get_sensitive_filter(self.emb_size)
        # self.filter_dict = nn.ModuleDict(
        #     {str(i + 1): get_sensitive_filter(self.emb_size) for i in range(self.filter_num)})
        


    def apply_filter(self, vectors):
        # if self.filter_mode == 'separate' and np.sum(filter_mask) != 0:
        #     filter_mask = np.asarray(filter_mask)
        #     idx = filter_mask.dot(2**np.arange(filter_mask.size))
        #     sens_filter = self.filter_dict[str(idx)]
        #     result = sens_filter(vectors)
        # elif self.filter_mode == 'combine' and np.sum(filter_mask) != 0:
        #     result = None
        #     for idx, val in enumerate(filter_mask):
        #         if val != 0:
        #             sens_filter = self.filter_dict[str(idx + 1)]
        #             result = sens_filter(vectors) if result is None else result + sens_filter(vectors)
        #     result = result / np.sum(filter_mask)   # average the embedding
        # else:
        #     result = vectors
        #sens_filter = self.filter_dict[str(1)]

        result = self.sens_filter(vectors)

        return result

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


    def get_relevances(self, model, user, items):
        pred_eval = model.model_(user, items)

        return pred_eval.cpu().data.numpy()
    
    def loss(self, predictions: torch.Tensor, current, data, reduction:str = 'mean') -> torch.Tensor:
        
        if 'occur' in self.dflag:
            u_occur_time = current['u_occur_time']
            i_occur_time = current['i_occur_time']
        if 'degree' in self.dflag:
            u_degree_weight = current['u_degree_weight']


        pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:1+self.num_neg] # 1 pos : self.num_neg neg
        loss = -(pos_pred[:,None] - neg_pred).sigmoid().log().mean(dim=1)

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

        return loss

    # def forward(self, feed_dict, filter_mask):
    #     out_dict = self.predict(feed_dict, filter_mask)
    #     batch_size = int(feed_dict[LABEL].shape[0] / 2)
    #     pos, neg = out_dict['prediction'][:batch_size], out_dict['prediction'][batch_size:]
    #     loss = -(pos - neg).sigmoid().log().sum()
    #     out_dict['loss'] = loss
    #     return out_dict

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
            #self.buffer = self.model.buffer and self.phase != 'train'

            self.train_boundary = corpus.n_train_batches
            
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

            self._set_user_item_occurrence_time()
            self._set_user_item_historical_degree()
            #self.decay_factor = 1e-3
            self.decay_factor = args.cay
            self.dflag = args.dflag

            #decay_mat = np.exp(self.decay_factor * (his_time_mat - cur_time))

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