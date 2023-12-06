# -*- coding: UTF-8 -*-

import os
import sys
import pickle
import logging
import argparse
import numpy as np
import torch
import copy
import pandas as pd

from helpers import MetaReader, MetaRunner
from models import  MetaModel
from models.general import BPR, NCF
from utils import utils
import re


def parse_global_args(parser):
    parser.add_argument('--gpu', type=str, default='0',
                        help='Set CUDA_VISIBLE_DEVICES')
    parser.add_argument('--verbose', type=int, default=logging.INFO,
                        help='Logging Level, 0, 10, ..., 50')
    parser.add_argument('--log_file', type=str, default='',
                        help='Logging file path')
    parser.add_argument('--result_file', type=str, default='',
                        help='Result file path')
    parser.add_argument('--result_folder', type=str, default='',
                        help='Result folder path')
    parser.add_argument('--random_seed', type=int, default=2021,
                        help='Random seed of numpy and pytorch.')
    parser.add_argument('--load', type=int, default=0,
                        help='Whether load model and continue to train')
    parser.add_argument('--train', type=int, default=1,
                        help='To train the model or not.')
    parser.add_argument('--finetune', type=int, default=0,
                        help='To finetune the model or not.')
    parser.add_argument('--eval', type=int, default=0,
                        help='To evaluate the model or not.')
    parser.add_argument('--time', type=str, default='',
                        help='Time to index model')
    parser.add_argument('--regenerate', type=int, default=0,
                        help='Whether to regenerate intermediate files.')
    parser.add_argument('--message', type=str, default='',
                        help='Additional message to add on the log/model name.')
    parser.add_argument('--dyn_update', type=int, default=0,
                        help='dynamic update strategy.')
    return parser


def main():
    logging.info('-' * 45 + ' BEGIN: ' + utils.get_time() + ' ' + '-' * 45)
    exclude = ['check_epoch', 'log_file', 'model_path', 'path', 'pin_memory',
               'regenerate', 'sep', 'train', 'verbose']
    #logging.info(utils.format_arg_str(args, exclude_lst=exclude))

    # Random seed
    utils.fix_seed(args.random_seed)

    # GPU
    #os.environ["CUDA_VISIBLE_DEVICES"] = 'cpu'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logging.info('cuda available: {}'.format(torch.cuda.is_available()))
    logging.info('cuda device: {}'.format(args.gpu))
    # logging.info('cuda available: {}'.format(torch.cuda.is_available()))
    # logging.info('# cuda devices: {}'.format(torch.cuda.device_count()))

    # Read data
    # corpus_path = os.path.join(args.path, args.dataset, model_name.reader + '.pkl')
    corpus_path = os.path.join(args.path, args.dataset, args.suffix, args.s_fname, model_name.reader + '.pkl')
    
    if not args.regenerate and os.path.exists(corpus_path):
        logging.info('Load corpus from {}'.format(corpus_path))
        corpus = pickle.load(open(corpus_path, 'rb'))
        #logging.info('Corpus loaded')
    else:
        corpus = reader_name(args)
        logging.info('Save corpus to {}'.format(corpus_path))
        pickle.dump(corpus, open(corpus_path, 'wb'))

    args.keys = ['train', 'test']
    # corpus.n_train = int(corpus.n_batches*args.train_ratio)*args.batch_size
    # corpus.n_test = corpus.dataset_size - corpus.n_train
    logging.info('Total instances: {}'.format(corpus.dataset_size))
    logging.info('Train instances: {}'.format(corpus.n_train_batches*args.batch_size))
    logging.info('Test instances: {}'.format(corpus.n_test_batches*args.batch_size))
    # for snapshot_idx in range(corpus.n_snapshots):
    #     snap_boundaries.append(snapshot_idx * int(corpus.n_test_batches/corpus.n_snapshots) * corpus.batch_size)
    logging.info('Snap boundaries: {}'.format(corpus.snap_boundaries))


    # # Define model (e.g., BPR or NCF)
    # model = model_name(args, corpus)
    # #logging.info(model)
    # model.apply(model.init_weights)
    # #model.actions_before_train()
    # model.to(model._device)

    # Run model 
    runner = runner_name(args, corpus)
    data_dict = dict()
    # pre-train
    # if args.dyn_update == -2:
    force_train = False

    # full-retraining   
    if 'fulltrain' in args.dyn_method:
        phase = 'fulltrain'
        time_d = {}
        for idx, n_idx in enumerate(corpus.snap_boundaries):
            utils.fix_seed(args.random_seed)
            model = model_name(args, corpus)
            model.apply(model.init_weights)
            model.to(model._device)
            data_dict['train'] = model_name.Dataset(model, args, corpus, phase, n_idx)

            if os.path.exists(model.model_path+'_snap{}'.format(corpus.n_snapshots-1)):
                args.train = 0

            if args.train > 0 or force_train:
                t = runner.train(model, data_dict, args, snap_idx=idx)
                time_d['period_{}'.format(idx)] = t

            #print('snap_idx: {}'.format(idx))
        if args.train > 0:
            with open(args.test_result_file+'_time_test', 'w+') as f:
                for k, v in time_d.items():
                    f.writelines('{}\t'.format(k))
                f.writelines('\n')
                for k, v in time_d.items():
                    f.writelines('{:.4f}\t'.format(v))
                f.writelines('\n')
                for k, v in time_d.items():
                    f.writelines('{:.4f}\t'.format(v/60))
    
    elif 'modi-fine' in args.dyn_method:
        utils.fix_seed(args.random_seed)
        model = model_name(args, corpus)
        model.apply(model.init_weights)
        model.to(model._device)

        snap_thres = re.sub('[^0-9]', '', args.dyn_method)
        if snap_thres == '':
            snap_thres = -1 # default
            logging.info('WARNING: no snapshot threshold specified, using default value -1')
        else:
            snap_thres = int(snap_thres)
        n_idx = corpus.snap_boundaries[snap_thres]
        data_dict['train'] = model_name.Dataset(model, args, corpus, 'fulltrain', n_idx)
        data_dict['test'] = model_name.Dataset(model, args, corpus, 'test')

        if os.path.exists(model.model_path+'_snap{}'.format(corpus.n_snapshots-1)):
            args.train = 0

        if args.train > 0 or force_train:
            t = runner.train(model, data_dict, args, snap_idx=snap_thres)

    # 'finetune' or 'pretrain'
    else:
        utils.fix_seed(args.random_seed)
        model = model_name(args, corpus)
        model.apply(model.init_weights)
        model.to(model._device)
        
        for phase in ['train', 'test']:
            data_dict[phase] = model_name.Dataset(model, args, corpus, phase)
            # data_dict[phase].g = copy.deepcopy(corpus.g.to(model._device))

        #print(model.model_path+'_snap{}'.format(corpus.n_snapshots-1))
        if os.path.exists(model.model_path+'_snap{}'.format(corpus.n_snapshots-1)):
            args.train = 0

        if args.train > 0 or force_train:
            runner.train(model, data_dict, args)
    #logging.info(os.linesep + 'Test After Training: ' + runner.print_res(model, data_dict, args, meta_model))

    utils.fix_seed(args.random_seed)
    tester = tester_name(args, corpus)
    tester.dp(args, model)

    # model.actions_after_train()
    logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)

def post():
    return args.test_result_file

if __name__ == '__main__':
    init_parser = argparse.ArgumentParser(description='Model')
    init_parser.add_argument('--model_name', type=str, default='BPR', help='Choose a model to run.')
    init_parser.add_argument('--dyn_method', type=str, default='default', help='Choose a model to run.')
    init_args, init_extras = init_parser.parse_known_args()
    #print(init_args.model_name)
    model_name = eval('{0}.{0}'.format(init_args.model_name))
    # if not init_args.meta_name == 'default':
    #     meta_name = eval('{0}.{0}'.format(init_args.meta_name))
    reader_name = eval('{0}.{0}'.format(model_name.reader))
    runner_name = eval('{0}.{0}'.format(model_name.runner))
    tester_name = eval('{}.{}'.format('MetaRunner','Tester'))
    # Args
    parser = argparse.ArgumentParser(description='')
    parser = parse_global_args(parser)
    parser = reader_name.parse_data_args(parser)
    parser = runner_name.parse_runner_args(parser)
    parser = model_name.parse_model_args(parser)
    parser = tester_name.parse_tester_args(parser)
    # if not init_args.meta_name == 'default':
    #     parser = meta_name.parse_meta_args(parser)
    args, extras = parser.parse_known_args()

    if init_args.dyn_method == 'finetune':
        pass
    elif 'fulltrain' in init_args.dyn_method:
        args.tepoch = -1
    elif 'pretrain' in init_args.dyn_method:
        args.tepoch = -1

    if args.DRM == 'none':
        args.tau = -1
        args.num_neg_fair = -1
        args.DRM_weight = 1.0
        print('when DRM is none: no fairness reg')
    # Logging configuration
    #log_args = [init_args.model_name, args.dataset, str(args.random_seed)]
    if args.dflag == 'default':
        args.dflag = 'none'
        args.DRM = 'none'
        args.DRM_weight = 1.0
        args.tau = -1
        args.num_neg_fair = -1
        print('when dflag is default: no fairness reg')


    args.s_fname = '{}_{}_{}_s{}'.format(args.split_type, args.train_ratio, args.batch_size, args.n_snapshots)
    #log_args = [init_args.model_name, args.dataset, args.suffix, args.s_fname] # + str(args.test_length)]
    log_args1 = [args.dataset, args.s_fname, init_args.dyn_method] # + str(args.test_length)]
    log_args2 = []
    if args.random_seed != 2021:
        log_args1.append('random_seed=' + str(args.random_seed))

    log_args = [args.dataset, args.s_fname, init_args.dyn_method]
    if 'adver' in init_args.model_name:
        params = ['lr','l2','epoch', 'tepoch', 'num_neg', 'reg_weight', 'd_steps']

    else:
        if args.DRM =='none':
            params = ['lr','l2','epoch', 'tepoch', 'num_neg', 'random_seed']
        else:
            # if args.DRM_weight == 1.0:
            #     params = ['lr','l2','epoch', 'tepoch', 'num_neg', 'num_neg_fair', 'DRM', 'tau']
            # else:
            params = ['lr','l2','epoch', 'tepoch', 'num_neg','num_neg_fair', 'DRM', 'DRM_weight', 'tau']
            if args.DRM_weight < 0:
                params.append('thres')

        if args.dflag != 'none':
            params += ['dflag', 'cay']

    for arg in params:
        log_args2.append(arg + '=' + str(eval('args.' + arg)))
        #log_args.append(arg + '=' + str(eval('args.' + arg)))


    log_file_name1 = '__'.join(log_args1).replace(' ', '__')
    log_file_name2 = '__'.join(log_args2).replace(' ', '__')
    # log_file_name = '__'.join([log_file_name, init_args.dyn_method]).replace(' ', '__')
    #log_file_name = '__'.join(log_args).replace(' ', '__')

    ### for test
    folder_name = init_args.model_name #+ '_230528'

    # if 'test' in args.message:
    #     folder_name += '_230528'

    if args.model_path == '':
        args.model_path = '../model/{}/{}/{}/{}'.format(folder_name, log_file_name1, log_file_name2, init_args.dyn_method)
        #args.meta_model_path = '../model/{}/{}/{}_meta'.format(init_args.model_name, log_file_name, init_args.dyn_method)
    utils.check_dir(args.model_path)

    if args.log_file == '':
        args.log_file = '../log/{}/{}/{}.txt'.format(folder_name, log_file_name1, log_file_name2)
    utils.check_dir(args.log_file)

    if args.test_result_file == '':
        args.test_result_file = '../test_result/{}/{}/{}/'.format(folder_name, log_file_name1, log_file_name2)
    utils.check_dir(args.test_result_file)
    

    args.dyn_method = init_args.dyn_method
    args.model_name = init_args.model_name
    
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    #logging.basicConfig(filename=args.log_file)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    #logging.info(init_args)
    logging.info(log_file_name1+'__'+log_file_name2)
    # logging.flush()
    # logging.get_absl_handler().use_absl_log_file(self.exp_name + '.log', self.log_path)

    main()