# FADE
The repository is the implementation of FADE: FAir Dynamic rEcommender
10.5281/zenodo.10669097

The Online Appendix is available at: https://sites.google.com/view/fade-www24/home.



## Requirements
- python 37
- pytorch v1.13.0
- scikit-learn
- tqdm


## Usage

You can use directly the following command to run the FADE and save the result file in "test_result" folder.

```bash
cd src
python python main.py --dataset 'Movielenz' --model_name 'BPR' --dyn_model 'finetune' --tepoch '10' --num_neg '4' --num_neg_fair '4' --lr '0.001' --l2 '1e-04' --DRM 'log-onlypos' --DRM_weight 1.0 --tau 3.0 --train_ratio 0.6 --batch_size 256 --random_seed 2021 --gpu 0
```
- dataset: 'Movielenz', 'Modcloth'
- model_name: the backbone recommendation model ('BPR' (MF-BPR), 'NCF' (NCF-BPR))
- dyn_model: dynamic model update strategy ('finetune' (fine-tuning), 'fulltrain' (retraining), 'pretrain' (pretraining))
- tepoch: the number of epoch of dynamic update phase
- num_neg: the number of negative items for BPR recommendation loss
- num_neg_fair: the number of negative items for the fairness loss
- lr: learning rate
- l2: l2 regularization
- DRM: the type of fairness loss ('log': L_{fair}, 'absolute': L_{fair-abs})
- DRM_weight: \lambda (scaling parameter of fairness loss)
- tau: \tau (the temperature parameter in the relaxed permutation matrix)
- train_ratio: the ratio of pre-training data of the entire dataset
- batch_size
- random_seed
- gpu: gpu number

You can also use the "_tester.py" to run FADE with the script with user-specified hyperparameters.
```bash
python _tester.py
```
In "_tester.py", you can change the hyperparameters of the FADE and dataset pre-processing.

In "data" folder, two datasets used in the paper are avaliable. 



python main.py --dataset 'Movielenz' --model_name 'BPR' --dyn_model 'finetune' --tepoch '10' --num_neg '4' --num_neg_fair '4' --lr '0.001' --l2 '1e-04' --DRM 'log-onlypos' --DRM_weight 1.0 --tau 3.0 --batch_size 256 --random_seed 2021 --train_ratio 0.6

