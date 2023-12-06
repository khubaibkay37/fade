# -*- coding: UTF-8 -*-

import argparse
import os
import time
import pickle
import json
import logging
import math
import copy
import torch
import sys
import dgl
from random import randint, choice

import scipy.sparse
import pandas as pd
import numpy as np
import datetime

from utils import utils
from torch.nn.utils.rnn import pad_sequence
from torch import nn


class MetaReader(object):
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--path', type=str, default='../data/',
                            help='Input data dir.')
        parser.add_argument('--suffix', type=str, default='fade',
                            help='Input data dir.')
        parser.add_argument('--dataset', type=str, default='',
                            help='Choose a dataset.')
        parser.add_argument('--sep', type=str, default='\t',
                            help='Sep of csv file.')
        parser.add_argument('--train_ratio', type=float, default=0.8,
                            help='Ratio of the train dataset')
        parser.add_argument('--duplicate', type=int, default=1,
                            help='Coalesce duplicate elements in adjacency matrix')
        parser.add_argument('--fname', type=str, default='freq',
                            help='Freq (> 20 records) or whole')
        parser.add_argument('--s_fname', type=str, default='',
                            help='Specific data folder name')
        parser.add_argument('--n_snapshots', type=int, default=10,
                            help='Number of test snapshots')
        parser.add_argument('--split_type', type=str, default='size',
                            help='Data split type')


        return parser


    def __init__(self, args):
        self.sep = args.sep
        self.prefix = args.path
        self.suffix = args.suffix
        self.dataset = args.dataset
        self.train_ratio = args.train_ratio
        self.batch_size = args.batch_size
        self.fname = args.fname
        self.s_fname = args.s_fname
        self.random_seed = args.random_seed
        self.n_snapshots = args.n_snapshots 
        self.split_type = args.split_type

        t0 = time.time()
        self._read_data()
        #print(self.data_df['user_id'].max(),len(self.data_df['user_id'].unique()))
        #print(self.data_df['item_id'].max(),len(self.data_df['item_id'].unique()))

        #logging.info('Counting dataset statistics...')
        self.n_users, self.n_items = self.data_df['user_id'].max()+1, self.data_df['item_id'].max()+1
        self.dataset_size = len(self.data_df)
        self.n_batches = math.ceil(self.dataset_size/self.batch_size)
        logging.info('"# user": {}, "# item": {}, "# entry": {}'.format(self.n_users, self.n_items, self.dataset_size))
        path = os.path.join(self.prefix, self.dataset, self.suffix, self.s_fname)
        if not os.path.exists(path):
            os.mkdir(path)
        del path

        self._set_snap_boundaries()
        self._save_snapshot_files()

        #logging.info('Saving data into mini-batch pickle files')
        self.user_list = self.data_df['user_id'].to_numpy()
        self._save_user_clicked_set()
        self._save_hist_set()
        self._save_mini_batch()
        # self._save_last_batch()
        # self._save_heterograph()


        self.user_attr_path = os.path.join(self.prefix, self.dataset, self.suffix, 'user_attr')


        del self.df
        # del self.data_df
        logging.info('Done! [{:<.2f} s]'.format(time.time() - t0) + os.linesep)


    def _set_snap_boundaries(self):
        if 'size' in self.split_type:
            # Split in equal size
            self.n_train_batches = int(self.n_batches*self.train_ratio)
            self.n_test_batches = self.n_batches - self.n_train_batches
            self.n_batches_per_snapshot = int(self.n_test_batches/self.n_snapshots)
            self.snap_boundaries = []
            for snapshot_idx in range(self.n_snapshots):
                self.snap_boundaries.append(snapshot_idx * self.n_batches_per_snapshot)

        elif 'time' in self.split_type:
            data = self.df.values.astype(np.int64)
            date = []
            for d in data:
                t = datetime.datetime.fromtimestamp(int(d[2])).timetuple()
                date.append([t[0],t[1]])

            # Split input data into snapshots divided by the pre-defined number of months
            prev_month = -1
            snapshots = {}

            k = -1
            threshold = threshold_cnt = self.n_snapshots # within how many months/years?
            for d in date:
                # month
                if d[1] == prev_month:
                    snapshots[k].append(d)
                elif threshold_cnt < threshold:
                    threshold_cnt += 1
                    snapshots[k].append(d)
                    prev_month = d[1]
                else:
                    k += 1
                    snapshots[k] = []
                    snapshots[k].append(d)
                    prev_month = d[1]
                    threshold_cnt = 1

            accum_snapshots = {}
            for k, snap in snapshots.items():
                # 0,1,..,n-1,n
                accum_snapshots[k] = []
                for i in range(k+1):
                    accum_snapshots[k].extend(snapshots[i])
                    
            snap_boundaries = []
            for k, snap in accum_snapshots.items():
                snap_boundaries.append(round(len(snap)/self.batch_size))

            # Here, self.train_ratio is the number of time periods for the training data
            print(self.train_ratio)
            self.n_train_batches = snap_boundaries[int(self.train_ratio)-1]
            self.n_test_batches = self.n_batches - self.n_train_batches

            print('ori_snap_boundaries: {}'.format(snap_boundaries))

            snap_boundaries = snap_boundaries[int(self.train_ratio):-1]
            for i, _ in enumerate(snap_boundaries):
                snap_boundaries[i] -= self.n_train_batches
            self.snap_boundaries = snap_boundaries

            print('snap_boundaries: {}'.format(self.snap_boundaries))


    def _save_snapshot_files(self):
        self.snapshots_path = os.path.join(self.prefix, self.dataset, self.suffix, self.s_fname, 'snapshots')
        if not os.path.exists(self.snapshots_path):
            os.mkdir(self.snapshots_path)

        #test_settings = ['remain', 'fixed', 'next']

        for idx, snap_boundary in enumerate(self.snap_boundaries):
            snapshot_train = self.data_df[:(self.n_train_batches + snap_boundary) * self.batch_size].values.astype(np.int64)

            if idx == 0:
                gap = 0
            else:
                gap = self.snap_boundaries[idx] - self.snap_boundaries[idx-1]
            snapshot_train_new = self.data_df[(self.n_train_batches + gap) * self.batch_size:(self.n_train_batches + snap_boundary) * self.batch_size].values.astype(np.int64)

            snapshot_test = self.data_df[(self.n_train_batches + snap_boundary) * self.batch_size:].values.astype(np.int64)
            utils.write_interactions_to_file(os.path.join(self.snapshots_path, 'remain_train_snap'+str(idx)), snapshot_train)
            utils.write_interactions_to_file(os.path.join(self.snapshots_path, 'remain_test_snap'+str(idx)), snapshot_test)
            #utils.write_interactions_to_file(os.path.join(self.snapshots_path, 'remain_train_new_snap'+str(idx)), snapshot_train_new)

            #snapshot_train = self.data_df[:(self.n_train_batches + snap_boundary) * self.batch_size].values.astype(np.int64)
            snapshot_test = self.data_df[(self.n_train_batches + self.snap_boundaries[-1]) * self.batch_size:].values.astype(np.int64)
            utils.write_interactions_to_file(os.path.join(self.snapshots_path, 'fixed_train_snap'+str(idx)), snapshot_train)
            utils.write_interactions_to_file(os.path.join(self.snapshots_path, 'fixed_test_snap'+str(idx)), snapshot_test)
            #utils.write_interactions_to_file(os.path.join(self.snapshots_path, 'fixed_train_new_snap'+str(idx)), snapshot_train_new)

            #snapshot_train = self.data_df[:(self.n_train_batches + snap_boundary) * self.batch_size].values.astype(np.int64)
            if idx == len(self.snap_boundaries)-1:
                snapshot_test = self.data_df[(self.n_train_batches + snap_boundary) * self.batch_size:].values.astype(np.int64)
            else:
                gap = self.snap_boundaries[idx+1] - self.snap_boundaries[idx]
                snapshot_test = self.data_df[(self.n_train_batches + snap_boundary) * self.batch_size:(self.n_train_batches + snap_boundary + gap) * self.batch_size].values.astype(np.int64)
            utils.write_interactions_to_file(os.path.join(self.snapshots_path, 'next_train_snap'+str(idx)), snapshot_train)
            utils.write_interactions_to_file(os.path.join(self.snapshots_path, 'next_test_snap'+str(idx)), snapshot_test)
            #utils.write_interactions_to_file(os.path.join(self.snapshots_path, 'next_train_new_snap'+str(idx)), snapshot_train_new)


    def _read_data(self):
        logging.info('Reading data from \"{}\", dataset = \"{}\", suffix = \"{}\", fname = \"{}\" '.format(self.prefix, self.dataset, self.suffix, self.fname))
        self.df = pd.read_csv(os.path.join(self.prefix, self.dataset, self.suffix, self.fname +'.csv'), sep=self.sep)  # Let the main runner decide the ratio of train/test
        self.data_df = self.df.loc[:, ['user_id', 'item_id']]#.values.astype(np.int64) # (number of items, 2)


    def _save_user_clicked_set(self):
        user_clicked_set_path = os.path.join(self.prefix, self.dataset, self.suffix, self.s_fname, 'user_clicked_set.txt')
        logging.info('Load user_clicked_set')

        try:
            self.user_clicked_set = pickle.load(open(user_clicked_set_path, 'rb'))
            logging.info("Successfully loaded saved user_clicked_set")
        except FileNotFoundError as e:
            logging.info('File not found, create user_clicked_set')
            self.user_clicked_set = self.data_df.groupby(['user_id'])['item_id'].unique().to_dict()
            pickle.dump(self.user_clicked_set, open(user_clicked_set_path, 'wb'))
            logging.info('Saved user_clicked_set')


    def _save_hist_set(self):
        '''Prepare previous interactions of users and items during offline meta-model training.
        '''
        item_hist_set_path = os.path.join(self.prefix, self.dataset, self.suffix, self.s_fname, 'item_hist_set.txt')
        user_hist_set_path = os.path.join(self.prefix, self.dataset, self.suffix, self.s_fname, 'user_hist_set.txt')
        logging.info('Load hist_set')

        try:
            self.item_hist_set = pickle.load(open(item_hist_set_path, 'rb'))
            self.user_hist_set = pickle.load(open(user_hist_set_path, 'rb'))
            logging.info("Successfully loaded saved hist_set")
        except FileNotFoundError as e:
            logging.info('File not found, create hist_set')
            train_df = self.data_df.loc[:self.n_train_batches*self.batch_size,:]
            self.item_hist_set = train_df.groupby(['item_id'])['user_id'].unique().to_dict()
            self.user_hist_set = train_df.groupby(['user_id'])['item_id'].unique().to_dict()
            pickle.dump(self.item_hist_set, open(item_hist_set_path, 'wb'))
            pickle.dump(self.user_hist_set, open(user_hist_set_path, 'wb'))
            logging.info('Saved hist_set')


    def _save_mini_batch(self):
        self.mini_batch_path = os.path.join(self.prefix, self.dataset, self.suffix, self.s_fname, 'mini_batch')
        if not os.path.exists(self.mini_batch_path):
            os.mkdir(self.mini_batch_path)

        for batch_idx in range(self.n_batches):
            ui_batch = torch.from_numpy(self.data_df[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size].values.astype(np.int64)) # (batch_size, 2)
            torch.save(ui_batch, open(os.path.join(self.mini_batch_path, str(batch_idx)+'.pt'), 'wb'))


    # def _save_last_batch(self):
    #     '''Save previous interaction of user and item in current mini-batch for online testing.
    #     Given user-item interaction (u, i) in current mini-batch, Prepare previous interaction of user and item

    #     The resulting user-item interactions (torch.LongTensor size of [batch_size*2, 2]) are saved in last_batch_path

    #         # different user, same item: (u', i), (u', -i)
    #         # for positive: get user, and pick another used item
    #         # for negative: get user, and pick random negative item

    #         # same user, different item: (u, i'), (u, -i')
    #         # for positive: get user, and pick another used item
    #         # for negative: get user, and pick random negative item

    #     Exception handling:
    #         If a user has no previous interactions, then the current item is selected.
    #         If an item has no previous interactions, then the current user is selected.
    #         -> Randomly sample previous user/item
    #     Prepare (torch.LongTensor size of [batch_size*2, 2])
    #     '''

    #     self.last_batch_path = os.path.join(self.prefix, self.dataset, self.suffix, self.s_fname, 'last_batch')
    #     if not os.path.exists(self.last_batch_path):
    #         os.mkdir(self.last_batch_path)

    #     test_start_batch_idx = self.n_train_batches

    #     # Training interactions -> derive last interaction in training data
    #     train_df = self.data_df.loc[:test_start_batch_idx*self.batch_size,:]

    #     ui_last = {u:None for u in range(self.n_users)}
    #     ui_last_ = train_df.groupby(['user_id'])['item_id'].unique().apply(lambda x: x[-1]).to_dict()
    #     ui_last.update(ui_last_)

    #     iu_last = {i:None for i in range(self.n_items)}
    #     iu_last_ = train_df.groupby(['item_id'])['user_id'].unique().apply(lambda x: x[-1]).to_dict()
    #     iu_last.update(iu_last_)

    #     # For each test mini-batch
    #     for batch_idx in range(test_start_batch_idx, self.n_batches):
    #         user_id, item_id = torch.from_numpy(self.data_df[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size].values.astype(np.int64)).T # (batch_size, 2)

    #         # same user, different item: (u, i'), (u, -i')
    #         # for positive: get user, and pick user's last item
    #         # for negative: get user, and pick random negative item
    #         pos_items_u = torch.zeros(size=(len(user_id), 1), dtype=torch.int64)
    #         neg_items_u = torch.zeros(size=(len(user_id), 1), dtype=torch.int64)
    #         for idx, user in enumerate(user_id):
    #             user_clicked_set = copy.deepcopy(self.user_clicked_set[user.item()])
    #             last_item = ui_last[user.item()]
    #             pos_items_u[idx] = last_item if last_item is not None else item_id[idx]
    #             neg_items_u[idx] = self._randint_w_exclude(user_clicked_set)
    #         items_u = torch.cat((pos_items_u, neg_items_u), axis=-1)

    #         # different user, same item: (u', i), (u', -i)
    #         # for positive: get item's last user, and pick item
    #         # for negative: get item's last user, and pick random negative item
    #         user_id_ = torch.zeros_like(user_id)
    #         for idx, item in enumerate(item_id): # pick u'
    #             last_user = iu_last[item.item()]
    #             user_id_[idx] = last_user if last_user is not None else user_id[idx]

    #         neg_items_u_ = torch.zeros(size=(len(user_id), 1), dtype=torch.int64)
    #         for idx, user in enumerate(user_id_):
    #             user_clicked_set = copy.deepcopy(self.user_clicked_set[user.item()])
    #             neg_items_u_[idx] = self._randint_w_exclude(user_clicked_set)
    #         items_u_ = torch.cat((item_id.reshape(-1, 1), neg_items_u_), axis=-1)

    #         # Update last interaction history
    #         for user, item in zip(user_id, item_id):
    #             ui_last[user.item()] = item.item()
    #             iu_last[item.item()] = user.item()

    #         # Save last interaction mini-batch
    #         user_id = torch.cat((user_id, user_id_), axis=0)
    #         item_id = torch.cat((items_u, items_u_), axis=0)

    #         # Pair of a user and a positive/neagative item
    #         last_batch = {'user_id': user_id, # [batch_size*2,]
    #                       'item_id': item_id} # [batch_size*2, 2]

    #         torch.save(last_batch, open(os.path.join(self.last_batch_path, str(batch_idx)+'.pt'), 'wb'))


    def _randint_w_exclude(self, clicked_set):
        randItem = randint(1, self.n_items-1)
        return self._randint_w_exclude(clicked_set) if randItem in clicked_set else randItem


    # def _save_heterograph(self):
    #     self.graph_path = os.path.join(self.prefix, self.dataset, self.suffix, self.s_fname, 'heterograph_train.bin')
    #     logging.info('Load heterograph_train')
    #     try:
    #         g_list, _ = dgl.load_graphs(self.graph_path,[0])
    #         self.g = g_list[0]
    #         logging.info("Successfully loaded heterograph_train")
    #     except Exception as e:
    #         logging.info('File not found, create heterograph_train')
    #         u, i = self.data_df[:self.n_train_batches*self.batch_size].values.T
    #         self.g = dgl.heterograph({('user', 'u_to_i', 'item'): (u, i),
    #                                   ('item', 'i_to_u', 'user'): (i, u)},
    #                                   num_nodes_dict={'user':self.n_users, 'item':self.n_items})
    #         dgl.save_graphs(self.graph_path, self.g)
    #         logging.info('Successfully saved heterograph_train')




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser = MetaReader.parse_data_args(parser)
    args, extras = parser.parse_known_args()

    args.path = '../../data/'
    corpus = MetaReader(args)

    corpus_path = os.path.join(args.path, args.dataset, 'mesia', 'Corpus.pkl')
    logging.info('Save corpus to {}'.format(corpus_path))
    pickle.dump(corpus, open(corpus_path, 'wb'))
