# -*- coding: utf-8 -*-
import os
import sys


dataset = ['Movielenz']
emb_algo = ['BPR']
dyn_models = ['finetune']
num_negs = [4]
num_negs_fair = [4]
DRMs = ['log-onlypos']
DRM_weight = [1.0]
taus = [3.0]
random_seeds = [2021]
n_snapshots = 10
split_type = 'size'
tepochs = [10]
gpu = 0
batch_sizes = [256]
lrs = [0.001]
l2s = ['1e-04']

os.chdir('src')

for random_seed in random_seeds:
	for data in dataset:
		if data == 'Modcloth':
			train_ratio = 0.7
		elif data == 'Movielenz':
			train_ratio = 0.6
		for model in emb_algo:
			for m in dyn_models:
				for DRM in DRMs:
					for drm_w in DRM_weight:
						for num_neg in num_negs:
							for num_neg_fair in num_negs_fair:
								for tepoch in tepochs:
									for tau in taus:
										for lr in lrs:
											for l2 in l2s:
												for batch_size in batch_sizes:
	
													arg = 'python main.py --model_name {} --gpu {} --tepoch {} --dataset {} --num_neg {} --num_neg_fair {}\
														--dyn_method {} --train_ratio {} --lr {} --l2 {} --DRM {} --DRM_weight {} --batch_size {} \
															--tau {} --n_snapshots {} --split_type {} --random_seed {}'.format(
														model, gpu, tepoch, data, num_neg, num_neg_fair, m, train_ratio, lr, l2, DRM, drm_w, batch_size, tau,
														n_snapshots, split_type, random_seed)

													print(arg)
													os.system(arg)


