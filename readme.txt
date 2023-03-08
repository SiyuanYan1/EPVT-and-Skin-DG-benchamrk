
###create environment and install packages

conda create -n env_name python=3.8 -y
conda activate env_name
pip install -r requirements.txt


###our core code of algorithm is in class DoPrompt_group_decompose in domainbed/algorithms.py, our training and testing code are in domainbed/scripts/train_epvt.py and
###domainbed/scripts/test_epvt.py

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

CUDA_VISIBLE_DEVICES=1  python -m domainbed.scripts.test_erm --model_nam