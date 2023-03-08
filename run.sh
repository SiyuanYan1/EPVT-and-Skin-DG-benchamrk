#!/bin/bash


###create domain groups based on artifacts
#
#python data_proc/grouping.py


### train our EPVT ###
#training isic2019

CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train_epvt --data_dir=./domainbed/data/ --steps 1501 --dataset SKIN --test_env 0 --algorithm DoPrompt_group_decompose --output_dir \
results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier": 1e-5, "prompt_dim":10}' --exp 'prompt_final_vis' --ood_vis True

##test EPVT on four ood datasets
CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.test_epvt --model_name 'prompt_final_vis.pkl'




#### train ERM baseline ###
CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_erm --data_dir=./domainbed/data/ --steps 1501 --dataset SKIN --test_env 0 --algorithm ERM \
--output_dir results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier":1e-5}' --exp 'erm_baseline'
##test ERM on four ood datasets

CUDA_VISIBLE_DEVICES=1  python -m domainbed.scripts.test_erm --model_name 'erm_baseline.pkl'