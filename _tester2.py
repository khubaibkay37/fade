# -*- coding: utf-8 -*-
import os
import sys


gen_dataset = ['Movielenz','Amazon','Modcloth','Kuairecbig']
gen_model = ['BPR', 'BPRbias', 'NCF', 'adverBPR', 'adverNCF','DMF']
dyn_model = ['finetune','fulltrain','pretrain','sliding']
# 'finetune1','finetune2','finetune3','finetune4','finetune5','finetune6','finetune7','finetune8'
# finetune# -> finetune starts from the #+1th snapshot
# fulltrain# -> pretrain with up to the #th snapshot, and then finetune from the #+1th snapshot
# fulltrain -> train from scratch
# pretrain -> train from the 0th snapshot (pretrain data)
# gen_test_types = ['LP-uniform','LP-mixed','LP-biased','sparsity_LP-uniform']
gen_algo = 'none'
gen_params = ['0.01']


dataset_idx = [1,3]
algo_idx = [2] 
# algo_idx = [2,3]



dyn_idx = [1,2,3]
dyn_idx = [1]


num_negs = [4]
num_negs_fair = [4]


# DRMs = ['log-onlypos', 'none']
# DRMs = ['log-onlypos'] # DH
# # DRMs = ['none'] # no fairness
# # DRMs = ['log-approx','log-neural','log-REC','log-listnet-typea','log-listnet-typeb']
# # DRMs = ['log-onlypos-time', 'none-time']
# DRMs = ['log-onlypos-time2','none-time2','log-approx-time','log-neural-time','log-REC-time','log-listnet-typea-time','log-listnet-typeb-time']
DRMs = ['log-onlypos-time', 'none-time']


DRM_weight = [1.0]
# #DRM_weight = [0.1,0.3,0.5,0.8,1.0,1.5,2.0,2.5,3.0,4.0]
# DRM_weight = [0.3,0.8,1.5,2.5,4.0]


# thresholds = [0.0,-0.01,-0.03,-0.05,-0.1,-0.2,-0.4,-0.8]
thresholds = [0.0]


taus = [1.0,3.0,5.0]
taus = [3.0]
# taus = [0.1,0.5,1.0,2.0,3.0,4.0,5.0]

# for approxNDCG alpha
# taus = [1,10,100]

# default = 2021
random_seeds = [2020,2019,2018,2017]
random_seeds = [2021]

message = 't'
#dflags = ['none','default','main-uoccur','main-fair-uoccur','main-udegree','main-fair-udegree','test-main-uoccur','test-main-fair-uoccur','test-main-udegree','test-main-fair-udegree']
# dflags = ['main-udegree','main-uoccur']
dflags = ['none']


# decay_factors = ['1e-1','1e-2','1e-4','1e-5']
decay_factors = ['1e-3']


n_snapshots = 10
split_type = 'size'


train = 2
#epochs = [200,300,400,500]
epochs = [100]


tepochs = [1,5,10,15,20,25,30,35,40,45,50]
# tepochs = [10]
tepochs = [10]

# lr = 0.001
# l2 = 0.000001

# lr = '1e-4'
gpu = 0
# batch_size = 256 ###########
#batch_sizes = [256, 512, 1024]
batch_sizes = [256]
# batch_sizes = [512, 1024]
# '1e-3', '1e-4', '5e-4'
#lrs = [0.001,0.0005,0.0001]
lrs = [0.001]

l2s = ['1e-04','1e-05','1e-06','1e-07','5e-08']
l2s = ['1e-04']

# reg_weights = [1,10,20,50]
reg_weights = [20]
# reg_weights = [1,10,50]
d_steps = [10]

#################
dataset = []
for i in dataset_idx:
	dataset.append(gen_dataset[i-1])
emb_algo = []
for i in algo_idx:
	emb_algo.append(gen_model[i-1])
dyn_models = []
for i in dyn_idx:
	dyn_models.append(dyn_model[i-1])
#################

# params_ = ['']

os.chdir('src')

for random_seed in random_seeds:
	for data in dataset:
		if data == 'Modcloth':
			train_ratios = [0.7]
		else:
			train_ratios = [0.6]
		for model in emb_algo:
			for m in dyn_models:
				for DRM in DRMs:
					if DRM == 'log-approx':
						taus_ = [1,10,100]
						taus_ = [10]
					else:
						taus_ = taus
					for drm_w in DRM_weight:
						for num_neg in num_negs:
							for num_neg_fair in num_negs_fair:
								for epoch in epochs:
									for tepoch in tepochs:
										for tau in taus_:
											for dflag in dflags:
												for cay in decay_factors:
													for reg in reg_weights:
														for d_step in d_steps:
															for lr in lrs:
																for l2 in l2s:
																	for batch_size in batch_sizes:
																		for train_ratio in train_ratios:
																			for thres in thresholds:
																
																				arg = 'python main.py --model_name {} --gpu {} --epoch {} --tepoch {} --dataset {} --num_neg {} --num_neg_fair {}\
																					--dyn_method {} --train_ratio {} --lr {} --l2 {} --DRM {} --DRM_weight {} --batch_size {} \
																						--tau {} --dflag {} --train {} --n_snapshots {} --split_type {} --cay {} --reg_weight {} --d_steps {} \
																						--message {} --thres {} --random_seed {}'.format(
																					model, gpu, epoch, tepoch, data, num_neg, num_neg_fair, m, train_ratio, lr, l2, DRM, drm_w, batch_size, tau, dflag, train, 
																					n_snapshots, split_type, cay, reg, d_step, message, thres, random_seed)

																				print(arg)
																				os.system(arg)


