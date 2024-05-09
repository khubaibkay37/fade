# FADE
The repository is the implementation of **Version 1.0** of FADE (FAir Dynamic rEcommender) proposed in the paper titled "Ensuring User-side Fairness in Dynamic Recommender Systems" (WWW'24): https://doi.org/10.1145/3589334.3645536.

**NOTE: There are Version 1.5 and Version 2.0 with the key updates below. For exact replication, one may refer to this repository. However, for future works, consider using higher versions for their advantages.**

### Key update in Version 1.5
* Flexibly using Dataloader during training with each data block, rather than relying on saved mini-batches on disk.

### Key update in Version 2.0
* Tests are based on Task-Next.
* Using GPU-based and All-ranking tests, ensuring consideration of all items in Top-K recommendation tasks.
* Using a validation set and early stop mechanism. Specifically, the first half of each subsequent data block is used for validation, while the second half is used for testing.


Full arXiv paper (including full Appendix): https://arxiv.org/pdf/2308.15651.

The Online Appendix: https://sites.google.com/view/fade-www24/home.

The offical code DOI: https://doi.org/10.5281/zenodo.10669096.


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

