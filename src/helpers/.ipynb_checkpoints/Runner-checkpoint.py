# -*- coding: UTF-8 -*-

import os
import gc
import copy
import torch
import logging
import numpy as np
import random
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List, NoReturn

from utils import utils
from models.Model import Model

import matplotlib.pyplot as plt


class Runner(object):
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument('--epoch', type=int, default=100,
                            help='Number of epochs.')
        parser.add_argument('--tepoch', type=int, default=10,
                            help='Number of epochs.')
        parser.add_argument('--lr', type=float, default=0.001,
                            help='Learning rate.')
        parser.add_argument('--l2', type=float, default=1e-04,
                            help='Weight decay in optimizer.')
        parser.add_argument('--batch_size', type=int, default=256,
                            help='Batch size during training.')
        parser.add_argument('--optimizer', type=str, default='Adam',
                            help='optimizer: GD, Adam, Adagrad, Adadelta')
        parser.add_argument('--num_workers', type=int, default=4,
                            help='Number of processors when prepare batches in DataLoader')
        parser.add_argument('--pin_memory', type=int, default=1,
                            help='pin_memory in DataLoader')

        return parser

    def __init__(self, args, corpus):
        self.epoch = args.epoch
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.l2 = args.l2
        self.optimizer_name = args.optimizer
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.result_file = args.result_file
        self.dyn_method = args.dyn_method
        self.time = None  # will store [start_time, last_step_time]

        self.snap_boundaries = corpus.snap_boundaries
        self.snapshots_path = corpus.snapshots_path
        self.test_result_file = args.test_result_file
        self.tepoch = args.tepoch
        self.DRM = args.DRM


    def _check_time(self, start=False):
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    def _build_optimizer(self, model):
        optimizer_name = self.optimizer_name.lower()
        if optimizer_name == 'adam':
            #logging.info("Optimizer: Adam")
            if 'parameters' in self.DRM:
                optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.l2)
            else:
                optimizer = torch.optim.Adam(
                model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2)
        else:
            raise ValueError("Unknown Optimizer: " + self.optimizer_name)
        return optimizer

    
    def make_plot(self, args, data, name, snap_idx=0):
        y = data
        x = range(len(y))
        plt.plot(x, y)
        plt.xlabel('epoch')
        plt.ylabel('{}'.format(name))
        plt.title('{}_{}'.format(name, snap_idx))
        plt.savefig(args.test_result_file+'_{}_{}.png'.format(name, snap_idx))
        plt.close()

    def train(self,
              model: torch.nn.Module,
              data_dict: Dict[str, Model.Dataset],
              args,
              snap_idx=0) -> NoReturn:

        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)

        self._check_time(start=True)
        self.time_d = {}
        fair_loss_list = list()

        for epoch in tqdm(range(self.epoch), ncols=100, mininterval=1):
            self._check_time()

            # if there is a pre-trained model, load it
            # if 'finetune' in self.dyn_method and os.path.exists(model.model_path+'_snap{}'.format(0)):
            #     print('Already trained: {}'.format(model.model_path+'_snap{}'.format(0)))
            #     model.load_model(add_path='_snap{}'.format(0))
            #     break

            loss, ori_loss, fair_loss, pd, flag = self.fit_offline(model, data_dict['train'])
            training_time = self._check_time()

            # Print first and last loss/test
            logging.info("Epoch {:<3} loss={:<.4f} ori_loss={:<.4f} fair_loss={:<.4f} [{:<.1f} s] ".format(
                            epoch + 1, loss, ori_loss, fair_loss, training_time))
            if flag:
                logging.info('NaN loss, stop training')
                break
            fair_loss_list.append(fair_loss)


        logging.info('dyn_method: {}'.format(self.dyn_method))
        # Full re-training
        if 'fulltrain' in self.dyn_method:
            model.save_model(add_path='_snap{}'.format(snap_idx))
            return self.time[1] - self.time[0]
        # pre-training
        elif 'pretrain' in self.dyn_method:
            for snap_idx in range(len(self.snap_boundaries)):
                model.save_model(add_path='_snap{}'.format(snap_idx))             
        # fine-tuning
        elif 'finetune' in self.dyn_method:
            model_ = copy.deepcopy(model) ###
            #model.save_model(add_path='_train') 

            self.time_d['pre-train'] = self.time[1] - self.time[0]
            model.save_model(add_path='_snap{}'.format(0))
            
            flag = self.dynamic_prediction(model_, data_dict['test'])
            with open(args.test_result_file+'_time_test.txt', 'w+') as f:
                for k, v in self.time_d.items():
                    f.writelines('{}\t'.format(k))
                f.writelines('\n')
                for k, v in self.time_d.items():
                    f.writelines('{:.4f}\t'.format(v))
                f.writelines('\n')
                for k, v in self.time_d.items(): 
                    f.writelines('{:.4f}\t'.format(v/60))

        logging.info(os.linesep + "[{:<.1f} m] ".format((self.time[1] - self.time[0]) / 60))

    def fit_offline(self,
                    model: torch.nn.Module,
                    data: Model.Dataset) -> float:

        gc.collect()
        torch.cuda.empty_cache()

        loss_lst, ori_loss_lst, fair_loss_lst = list(), list(), list()
        pd_list = list()
        dl = DataLoader(data, batch_size=1, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)
        
        #for current in tqdm(dl, leave=True, desc='Epoch {:<3}'.format(epoch), ncols=100, mininterval=1):
        flag = 0
        for current in dl:
            # print("In runner: ", current['attr'])
            current = utils.batch_to_gpu(utils.squeeze_dict(current), model._device)
            current['batch_size'] = len(current['user_id'])
            # print("In runner: ", current.keys())
            loss, prediction, ori_loss, fair_loss, pd = self.train_recommender_vanilla(model, current, data)

            loss_lst.append(loss)
            ori_loss_lst.append(ori_loss)
            if fair_loss is not None:
                fair_loss_lst.append(fair_loss)
            if pd is not None:
                pd_list.append(pd)


            flag = np.isnan(prediction).any()
            if flag: 
                break

        return np.mean(loss_lst).item(), np.mean(ori_loss_lst).item(), np.mean(fair_loss_lst).item(), np.mean(pd_list).item(), flag

    def dynamic_prediction(self,
                model: torch.nn.Module,
                data: Model.Dataset) -> float:

        self._check_time()

        gc.collect()
        torch.cuda.empty_cache()


        # explicitly collecting dynamic update data (load the saved mini-batches)
        starts = []
        ends = []
        for i in range(len(self.snap_boundaries)):
            if i == 0:
                starts.append(-100)
            else:
                starts.append(self.snap_boundaries[i-1])
            ends.append(self.snap_boundaries[i])

        snap_idx = 0
        data_custom = {}
        for start, end in zip(starts, ends):
            data_custom[snap_idx] = []
            data.fine_tune_snap_idx = snap_idx

            dl = DataLoader(data, batch_size=1, shuffle=False, num_workers=0, pin_memory=self.pin_memory)
            for i, current in enumerate(dl):
                if i < start:
                    continue
                if i >= end:
                    break

                current = utils.batch_to_gpu(utils.squeeze_dict(current), model._device)
                current['batch_size'] = len(current['user_id'])
                data_custom[snap_idx].append(current)
            #print(data_custom[snap_idx])
            snap_idx += 1

        t = self._check_time()
        logging.info('test batch collecting: {} s'.format(t))
        self.time_d['test batch collecting'] = t

        flag = 0
        for snap_idx, snapshot_data in data_custom.items():
            logging.info('snap_idx: {}'.format(snap_idx))

            
            # snap_idx == 0 -> pretrain data -> skip
            if snap_idx == 0:
                continue

            over_fair_loss_lst = list()
            over_pd_list = list()
            for e in tqdm(range(self.tepoch), desc='Until {:<3}'.format(ends[snap_idx])):
                gc.collect()
                torch.cuda.empty_cache()
                loss_lst = list()
                ori_loss_lst = list()
                fair_loss_lst = list()
                pd_list = list()
                for i, current in enumerate(snapshot_data):
                    loss, prediction, ori_loss, fair_loss, pd = self.train_recommender_vanilla(model, current, data)       
                    loss_lst.append(loss)
                    ori_loss_lst.append(ori_loss)
                    if fair_loss is not None:
                        fair_loss_lst.append(fair_loss)
                    if pd is not None:
                        pd_list.append(pd)

                    flag = np.isnan(prediction).any()
                    if flag: 
                        break

                logging.info("Epoch {:<3} loss={:<.4f} ori_loss={:<.4f} fair_loss={:<.4f} ".format(
                            e + 1, np.mean(loss_lst).item(), np.mean(ori_loss_lst).item(), np.mean(fair_loss_lst).item()))
                over_fair_loss_lst.append(np.mean(fair_loss_lst).item())
                over_pd_list.append(np.mean(pd_list).item())
                
                if flag:
                    logging.info('@@@ prediction contains invalid values @@@')
                    flag = 0
                    break

            model.save_model(add_path='_snap{}'.format(snap_idx))
            self.time_d['period_{}'.format(snap_idx)] = self._check_time()
       
        return flag


    def train_recommender_vanilla(self, model, current, data):
        # Train recommender
        model.train()
        # Get recommender's prediction and loss from the ``current'' data at t
        # print("In runnerr, len f current: ", len(current['user_id']), len(current['item_id']))
        
        prediction = model(current['user_id'], current['item_id'], self.DRM)
        loss, ori_loss, fair_loss, pd = model.loss(prediction, current, data, reduction='mean')

        # Update the recommender
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

        if fair_loss is not None:
            fair_loss = fair_loss.cpu().data.numpy()
        if pd is not None:
            pd = pd.cpu().data.numpy()

        return loss.cpu().data.numpy(), prediction.cpu().data.numpy(), ori_loss.cpu().data.numpy(), fair_loss, pd




# Top-K recommendation evaluation (CPU)
class Tester(object):
    @staticmethod
    def parse_tester_args(parser):
        parser.add_argument('--test_topk', type=str, default='[20]',
                            help='The number of items recommended to each user.')
        parser.add_argument('--test_metric', type=str, default='["ndcg1","f1"]',
                            help='["ndcg1","f1","recall"]')
        parser.add_argument('--test_result_file', type=str, default='',
                            help='')

        return parser


    def dp(self, args, model):

        
        if torch.cuda.is_available():
            model.to(model._device)

        # Test settings: 1. Task-R (predict the remaining interactions), 2. Task-N (live-stream (predict right next interactions)), 3. Task-fixed (predict the last time interactions)
        #test_settings = ['remain','next','fixed']
        test_settings = ['remain', 'next']

        for topk in self.topk:
            for setting in test_settings:
                for snap_idx in range(len(self.snap_boundaries)):
                    # if os.path.exists(os.path.join(self.test_result_file, '{}_snap{}.txt'.format(setting, len(self.snap_boundaries)-1))):
                    #     print('Already existing test files: {}'.format(os.path.join(self.test_result_file, '{}_snap{}.txt'.format(setting, len(self.snap_boundaries)-1))))
                    #     break

                    model.load_model(add_path='_snap{}'.format(snap_idx), flag=1)
                    model.eval()

                    train_file = os.path.join(self.snapshots_path, '{}_train_snap{}'.format(setting, snap_idx))
                    test_file = os.path.join(self.snapshots_path, '{}_test_snap{}'.format(setting, snap_idx))
                    result_str, info_str = self.recommendation(model, train_file, test_file, topk)

                    result_filename_ = os.path.join(self.test_result_file, '{}_{}_snap{}.txt'.format(topk, setting, snap_idx))
                    r_string = 'Top {} Results'.format(topk) + result_str #'\n\n\n\n' + info_str 
                    with open(result_filename_, 'w+') as f:
                        f.writelines(r_string)

                # mean values over snapshots
                d = {}
                for snap_idx in range(len(self.snap_boundaries)):
                    with open(os.path.join(self.test_result_file, '{}_{}_snap{}.txt'.format(topk, setting, snap_idx)), 'r') as f:
                        #lines = f.readlines()[1:4*len(self.metrics)+1+24] #len(self.metrics)
                        lines = f.readlines()[1:] #len(self.metrics)
                        data = [line.replace('\n','').split() for line in lines]

                        #cnt = 0
                        for value in data:
                            if d.get(value[0]) is None:
                                d[value[0]] = []
                            # if cnt >= 4*len(self.metrics):
                            #     d[value[0]].append(int(value[1]))
                            # else:
                            # Absolute PD
                            if True:
                                d[value[0]].append(abs(float(value[1])))
                            # PD
                            else:
                                d[value[0]].append(float(value[1]))
                            # cnt += 1

                start = 1
                end = 7
                # write mean values from start to end
                with open(os.path.join(self.test_result_file, '0_{}_mean_{}_from_t{}_to_t{}.txt'.format(topk, setting, start, end)), 'w+') as f:
                    cnt = 0
                    for k, v in d.items():
                        # if cnt == 4*3+1:
                        #     f.writelines('\n\n')
                        v = v[start:end]
                        f.writelines('{}\t{}\n'.format(k, sum(v)/len(v)))

                # Write trend from start to end
                with open(os.path.join(self.test_result_file, '0_{}_trend_{}_from_t{}_to_t{}.txt'.format(topk, setting, start, end)), 'w+') as f:
                    #cnt = 0
                    for k, v in d.items():
                        # cnt += 1
                        # if cnt == 4*3+1:
                        #     f.writelines('\n\n')
                        v = v[start:end]
                        f.writelines('{}'.format(k))
                        for v_ in v:
                            f.writelines('\t{}'.format(v_))
                        f.writelines('\n')



    def __init__(self, args, corpus):
        self.user_attr_file = corpus.user_attr_path
        self.snap_boundaries = corpus.snap_boundaries
        self.snapshots_path = corpus.snapshots_path
        self.num_neg_samples = 100
        if args.dataset == 'Modcloth':
            self.num_neg_samples = 100
        self.test_result_file = args.test_result_file

        self.topk = eval(args.test_topk)
        self.K = self.topk[0]
        self.metrics = [m.strip() for m in eval(args.test_metric)]
        #self.main_metric = '{}@{}'.format(self.metrics[0], self.topk[0])  # early stop based on main_metric

        print('Test start: topk: {}, metric: {}'.format(self.topk, self.metrics))

        # MovieLens-1M: i=0: gender, i=1: age, i=2: occupation
        binary = list(range(2)) # ['M'=0,'F'=1]

        if args.dataset == 'Modcloth':
            self.attr_type = ['body-shape']
        else:
            self.attr_type = ['genders']
        self.user_groups = [binary]

        #self.set_user_attr(train_file, user_attr_file, self.num_groups_degree)
        self.num_type_attr = 1
        

    def set_user_attr(self, train_file, user_attr_file):
        train_edges = utils.read_data_from_file_int(train_file)
        train_user_set, _ = utils.get_user_item_set(train_edges)

        # MovieLenz
        user_attr = utils.read_data_from_file_int(user_attr_file)
        user_attr_dict = {}
        for user in user_attr:
            #user_attr_dict[user[0]] = [user[1], user[2], user[3]]
            user_attr_dict[user[0]] = [user[1]]
        self.user_attr = {}

        for u_idx in train_user_set:
            self.user_attr[u_idx] = [user_attr_dict[u_idx][0]]

    def init_results(self):

        self.results = {k: 0.0 for k in self.metrics}
        self.num_test_users = 0

        self.results_user_attr = []
        self.num_users_per_group = []
        self.fairness_results = []
        for k in range(self.num_type_attr):
            self.results_user_attr.append({})
            self.num_users_per_group.append({})
            for attr in self.user_groups[k]:
                self.results_user_attr[k][attr] = {}
                for metric in self.metrics:
                    self.results_user_attr[k][attr][metric] = 0
                self.num_users_per_group[k][attr] = 0

        self.num_actual_users_per_group = copy.deepcopy(self.num_users_per_group)
        self.num_unseen_items_per_group = copy.deepcopy(self.num_users_per_group)
        self.num_test_pos_per_group = copy.deepcopy(self.num_users_per_group)
        self.num_train_pos_per_group = copy.deepcopy(self.num_users_per_group)
        

    def generate_recommendation_list_for_a_user(self, model, user, train_item_set, train_pos, test_pos, K, num_neg_samples=-1):

        #num_neg_samples = 100
        verbose = False
        # if user == 0:
        #   verbose = True

        new_item_set = list(set(train_item_set) - set(train_pos[user]) - set(test_pos[user]))

        if num_neg_samples != -1:
            if num_neg_samples > len(new_item_set):
                print(len(new_item_set))
                print('numer of train_item_set {}'.format(len(train_item_set)))
                print('number of items of the user {}'.format(len(train_pos[user])))
                print('number of test pos items of the user {}'.format(len(test_pos[user])))
            neg_samples = random.sample(new_item_set, num_neg_samples)
        else:
            neg_samples = new_item_set

        #print('check@@@ {}'.format(train_item_set+neg_samples))
        relevances = {}
        cnt = 0

        # In case of unseen test items, just use random embeddings of the model.
        # (n_items)
        # print("runnn")
        # print(len(neg_samples))
        # print(len(test_pos[user]))
        # print(neg_samples)
        candidate_items = test_pos[user] + neg_samples
        # print(user)

        user_ = torch.from_numpy(np.array(user))
        # print(len(candidate_items))
        candidate_items_ = torch.from_numpy(np.array(candidate_items))
        
        if torch.cuda.is_available():
            user_ = user_.to(model._device)
            candidate_items_ = candidate_items_.to(model._device)

        item_relevances = model.get_relevances(model, user_, candidate_items_)

        for item, relevance in zip(candidate_items, item_relevances):
            if item in train_item_set:
                relevances[item] = relevance
                #relevances[item] = train_embeddings[0][user] @ train_embeddings[1][item]
            else:
                relevances[item] = relevance
                cnt += 1

        sorted_relevances = sorted(relevances.items(), key=lambda x: x[1], reverse=True)

        if K > 0:
            recommendation_list = [rel[0] for rel in sorted_relevances][:K]
        else:
            recommendation_list = [rel[0] for rel in sorted_relevances]
        
        if verbose:
            print(recommendation_list)
            print('@@@CHECK length of rec list: {} = {}, no test item{}'.format(len(recommendation_list), K, cnt))

        return recommendation_list, cnt

    def measure_performance_for_a_user(self, user, recommendation_list, train_pos, test_pos, num_unseen_items):
        flag = 0

        for metric in self.metrics:

            if test_pos.get(user) is not None:

                if metric == 'recall':
                    value = self.measure_recall(recommendation_list, test_pos[user])
                elif metric == 'ndcg1':
                    value = self.measure_ndcg(recommendation_list, test_pos[user], method=1)
                elif metric == 'f1':
                    value = self.measure_f1(recommendation_list, test_pos[user])
                else:
                    raise ValueError('Undefined evaluation metric: {}.'.format(metric))

                self.results[metric] += value
                if flag == 0:
                    self.num_test_users += 1

                # For each of user groups that are divided by sensitive attributes
                for k in range(self.num_type_attr):
                    for attr in self.user_groups[k]:
                        if self.user_attr[user][k] == attr:
                            self.results_user_attr[k][attr][metric] += value
                            # Is the user in the test set or not

                            if flag == 0:
                                self.num_users_per_group[k][attr] += 1
                                self.num_unseen_items_per_group[k][attr] += num_unseen_items
                                self.num_test_pos_per_group[k][attr] += len(test_pos[user])
                                self.num_train_pos_per_group[k][attr] += len(train_pos[user])
                
                flag += 1
            else:
                print('@@@@@@@@@error@@@@@@@@@@')


    def recommendation(self, model, train_file, test_file, topk=20, num_neg_samples=-1):
        topk = topk
        num_neg_samples = self.num_neg_samples

        self.init_results()
        self.set_user_attr(train_file, self.user_attr_file)
        #self.num_type_attr = len(self.user_attr[0])

        # For each user, there are personalized items in the recommendation list and test positive items
        # K = max(topk)
        # print(train_file)
        train_edges = utils.read_data_from_file_int(train_file)
        test_edges = utils.read_data_from_file_int(test_file)
        train_pos = utils.get_user_dil_from_edgelist(train_edges)
        test_pos = utils.get_user_dil_from_edgelist(test_edges)
        train_user_set, train_item_set = utils.get_user_item_set(train_edges)
        test_user_set, test_item_set = utils.get_user_item_set(test_edges)


        # Do not test new users, which does not exist in the training set
        # Generate top-k recommendation list for each user
        # num_neg_samples = -1
        random.seed(10)
        # print(type(test_pos))
        # print(max(test_pos.keys()), min(test_pos.keys()))
        # print(train_pos, test_pos)
        for user in train_user_set:
            # Skip if the user is not in the test set
            if user in test_pos.keys():
                recommendation_list, num_unseen_items = self.generate_recommendation_list_for_a_user(model, user, train_item_set, train_pos, test_pos, topk, num_neg_samples)
                self.measure_performance_for_a_user(user, recommendation_list, train_pos, test_pos, num_unseen_items)

        # print(self.num_unseen_items_per_group)

        self.average_user()
        self.average_user_attr()
        self.count_info_per_group(train_user_set, self.num_actual_users_per_group)
        self.average_info_per_group(self.num_unseen_items_per_group)
        self.average_info_per_group(self.num_test_pos_per_group)
        self.average_info_per_group(self.num_train_pos_per_group)
        #num_actual_train_pos_per_group_total = self.get_average_out_degree_per_groups(self.num_actual_train_pos_per_group, self.num_actual_users_per_group, train_pos)

        info_str = ''
        # info_str = '@@@ User Groups @@@'
        # info_str += '\noverall_num_test_users: {}, overall_real_num_test_users: {}'.format(len(train_user_set), self.num_test_users)
        # # info_str += '\nThe number of unseen test items per group: {}'.format(self.num_unseen_items_per_group)
        # info_str += '\nThe number of test positive items per group: {}'.format(self.num_test_pos_per_group)
        # info_str += '\nThe number of (valid) users per group: {}'.format(self.num_users_per_group)
        # info_str += '\nThe number of (valid) train positive items per group: {}'.format(self.num_train_pos_per_group)
        # info_str += '\nThe number of actual users per group: {}'.format(self.num_actual_users_per_group)
        # info_str += '\nThe number of actual train positive items per group: {}'.format(self.num_actual_train_pos_per_group)


        result_str = self.get_results_str_()


        return result_str, info_str


    def get_results_str_(self):
        self.measure_unfairness()
        result_str = ''

        for k in range(self.num_type_attr):
            # if self.attr_type[k] == 'ages' or self.attr_type[k] == 'occupations':
            #     continue
            for metric in self.metrics:
                #result_str += '\n\n@@@ Unfairness values @@@'
                result_str += '\n{}\t{:.4f}'.format(metric+'__overall', self.results[metric])

                if self.binary_unfairness[metric].get(k) is not None:
                    result_str += '\n{}\t{:.4f}'.format(metric+'__'+self.attr_type[k], self.binary_unfairness[metric][k])
                    for attr in self.user_groups[k]:
                        result_str += '\n{}\t{:.4f}'.format(metric+'__'+str(attr), self.results_user_attr[k][attr][metric])


                else:
                    for attr in self.user_groups[k]:
                        result_str += '\n{}\t{:.4f}'.format(metric+'__'+str(attr), self.results_user_attr[k][attr][metric])

        return result_str

    def measure_unfairness(self):
        self.variance = {}
        self.binary_unfairness = {}
        for metric in self.metrics:
            self.variance[metric] = []
            self.binary_unfairness[metric] = {}
            for k in range(self.num_type_attr):
                value_list = []
                for attr in self.user_groups[k]:
                    #if self.results_user_attr[k][attr][metric] != 0:
                    value_list.append(self.results_user_attr[k][attr][metric])
                self.variance[metric].append(np.var(value_list))

                if len(value_list) == 2:
                    self.binary_unfairness[metric][k] = value_list[0] - value_list[1]
        #print(self.results_user_attr)

    def count_info_per_group(self, train_user_set, info):
        for i in train_user_set:
            for k in range(self.num_type_attr):
                for attr in self.user_groups[k]:
                    if self.user_attr[i][k] == attr:
                        #self.num_actual_users_per_group[k][attr] += 1
                        info[k][attr] += 1

    def average_info_per_group(self, info):
        for k in range(self.num_type_attr):
            for attr in self.user_groups[k]:
                if self.num_users_per_group[k][attr] == 0:
                    info[k][attr] = 0
                else:
                    info[k][attr] = round(info[k][attr] / self.num_users_per_group[k][attr])

    def average_user_attr(self):
        for k in range(self.num_type_attr):
            for attr in self.user_groups[k]:
                for metric in self.metrics:
                    if self.num_users_per_group[k][attr] == 0:
                        self.results_user_attr[k][attr][metric] = 0
                    else:
                        a = self.results_user_attr[k][attr][metric]
                        self.results_user_attr[k][attr][metric] /= self.num_users_per_group[k][attr]

    def average_user(self):
        for metric, value in self.results.items():
            self.results[metric] = value/self.num_test_users

    def measure_recall(self, rec_list, test_pos):
        hit_count = np.isin(rec_list, test_pos).sum()

        return hit_count / len(test_pos)
     
    def measure_num_hit(self, rec_list, test_pos):
        return np.isin(rec_list, test_pos).sum()

    # def measure_ndcg_deprecated(self, rec_list, test_pos):
    #     index = np.arange(len(rec_list))
    #     k = min(len(rec_list), len(test_pos))
    #     idcg = (1/np.log2(2+np.arange(k))).sum()
    #     dcg = (1/np.log2(2+index[np.isin(rec_list, test_pos)])).sum()

    #     return dcg/idcg

    def measure_ndcg(self, rec_list, test_pos, method=1):
        r = np.asfarray(np.isin(rec_list, test_pos))
        max_r = np.asfarray(sorted(np.isin(rec_list, test_pos), reverse=True))

        # if r.size:
        # dcg
        if method == 0:
            dcg = r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            dcg = np.sum(r / np.log2(np.arange(2, r.size + 2)))
        elif method == 2:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))

        # idcg
        if method == 0:
            idcg = max_r[0] + np.sum(max_r[1:] / np.log2(np.arange(2, max_r.size + 1)))
        elif method == 1:
            idcg = np.sum(max_r / np.log2(np.arange(2, max_r.size + 2)))
        if not idcg:
            return 0.
        else:
            return dcg / idcg

    def measure_hit_ratio(self, rec_list, test_pos):
        hit_count = np.isin(rec_list, test_pos).sum()

        if hit_count > 0:
            return 1.0
        else:
            return 0.0

    def measure_precision(self, rec_list, test_pos):
        hit_count = np.isin(rec_list, test_pos).sum()

        return hit_count / len(rec_list)

    def measure_average_precision(self, rec_list, test_pos, method=0):
        r = np.isin(rec_list, test_pos)
        out = [self.measure_precision(rec_list[:k+1], test_pos) for k in range(len(rec_list)) if r[k]]

        if not out:
            return 0.

        if method==0:
            return np.mean(out)
        elif method==1:
            return sum(out)/min(len(rec_list), len(test_pos))
        elif method==2:
            return sum(out)/len(test_pos)


    def measure_f1(self, rec_list, test_pos):
        recall = self.measure_recall(rec_list, test_pos)
        precision = self.measure_precision(rec_list, test_pos)
        if recall+precision == 0:
            value = 0
        else:
            value = 2*(recall*precision) / (recall+precision)

        return value

    def measure_mrr(self, rec_list, test_pos, method=0):

        r = np.asfarray(np.isin(rec_list, test_pos))

        if method == 0:
            rr = np.sum(r / np.arange(1, r.size + 1))   
        elif method == 1:
            r_ = 0
            for rank, i in enumerate(r):
                if i == 1:
                    r_ = rank + 1
                    break
            if r_ == 0:
                rr = 0
            else:
                rr = 1/r_

        return rr
