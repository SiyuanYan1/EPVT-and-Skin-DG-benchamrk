# EPVT: Environment-aware Prompt Vision Transformer for Domain Generalization in Skin Lesion Recognition<br><sub><sub>Official PyTorch implementation of the MICCAI 2023 paper</sub></sub>
[[`arXiv`](https://arxiv.org/pdf/2304.01508.pdf)]
[[`BibTex`](#citation)]
[[`MICCAI paper`](https://link.springer.com/chapter/10.1007/978-3-031-43990-2_24)]

## Introduction
[**EPVT: Environment-aware Prompt Vision Transformer for Domain Generalization in Skin Lesion Recognition**](https://arxiv.org/pdf/2304.01508.pdf)<br>
Siyuan Yan, Chi Liu, Zhen Yu, Lie Ju, Dwarikanath Mahapatra, Monika Janda, Peter Soyer, Zongyuan Ge


**[abstract]** *Skin lesion recognition using deep learning has made remarkable progress, and there is an increasing need for deploying these systems in real-world scenarios. However, recent research has revealed that deep neural networks for skin lesion recognition may overly depend on disease-irrelevant image artifacts (i.e. dark corners, dense hairs), leading to poor generalization in unseen environments. To address this issue, we propose a novel domain generalization method called EPVT, which involves embedding prompts into the vision transformer to collaboratively learn knowledge from diverse domains. Concretely, EPVT leverages a set of domain prompts, each of which plays as a domain expert, to capture domain-specific knowledge; and a shared prompt for general knowledge over the entire dataset. To facilitate knowledge sharing and the interaction of different prompts, we introduce a domain prompt generator that enables low-rank multiplicative updates between domain prompts and the shared prompt. A domain mixup strategy is additionally devised to reduce the co-occurring artifacts in each domain, which allows for more flexible decision margins and mitigates the issue of incorrectly assigned domain labels. Experiments on four out-of-distribution datasets and six different biased ISIC datasets demonstrate the superior generalization ability of EPVT in skin lesion recognition across various environments.*
![alt text](image/motivation.png)
## Method
<img src="image/method.png" alt="My Image" width="800">



## Installation
create the environment and install packages
```
conda create -n env_name python=3.8 -y
conda activate env_name
pip install -r requirements.txt
```

###our core code of algorithm is in class DoPrompt_group_decompose in domainbed/algorithms.py, our training and testing code are in domainbed/scripts/train_epvt.py and
###domainbed/scripts/test_epvt.py

## Preparing datasets

Download datasets
**ISIC2019**: download ISIC2019 training dataset from [here]([https://github.com/alinlab/BAR](https://challenge.isic-archive.com/data/#2019)

**Derm7pt**: download Derm7pt Clinical and Derm7pt Dermoscopic dataset from [here]([https://github.com/switchablenorms/CelebAMask-HQ](https://derm.cs.sfu.ca/Welcome.html))

**PH2**: download the PH2 dataset from [here]([https://github.com/fyu/lsun](https://www.fc.up.pt/addi/ph2%20database.html)

**PAD**: download the PAD-UFES-20 dataset from [here]([http://places2.csail.mit.edu/download.html](https://paperswithcode.com/dataset/pad-ufes-20)

Pre-processing the ISIC2019 dataset to construct the artifacts-based domain generalization training dataset, you need to modify paths in the pre-processing file accordingly.
```
python data_proc/grouping.py
```
## Directly accessing the processed datasets via GoogleDrive

All processed datasets are in [GoogleDrive](https://drive.google.com/file/d/12SoMs_44jD4mRT6JEyIfdjBa4Fw07i2m/view?usp=sharing)



## Training

Our benchmark is modified based on DomainBed, please refer to [DomainBed Readme](https://github.com/facebookresearch/DomainBed) for more details on commands running jobs. 

```sh
# Training EPVT on ISIC2019

CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train_epvt --data_dir=./domainbed/data/ --steps 1501 --dataset SKIN --test_env 0 --algorithm DoPrompt_group_decompose --output_dir \
results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier": 1e-5, "prompt_dim":10}' --exp 'prompt_final_vis' --ood_vis True

#Test EPVT on four OOD datasets
CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.test_epvt --model_name 'prompt_final_vis.pkl'

#Training ERM baseline on ISIC2019
CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_erm --data_dir=./domainbed/data/ --steps 1501 --dataset SKIN --test_env 0 --algorithm ERM \
--output_dir results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier":1e-5}' --exp 'erm_baseline'

#Test ERM on four ood datasets
CUDA_VISIBLE_DEVICES=1  python -m domainbed.scripts.test_erm --model_nam
```

## Requirements

```sh
pip install -r requirements.txt
```

## Citation

```bibtex
@article{zheng2022prompt,
  title={Prompt Vision Transformer for Domain Generalization},
  author={Zheng, Zangwei and Yue, Xiangyu and Wang, Kai and You, Yang},
  journal={arXiv preprint arXiv:2208.08914},
  year={2022}
}
```

## TODO

- Details about preparing the datasets.


## Citation

```bibtex
@misc{yan2023epvt,
      title={EPVT: Environment-aware Prompt Vision Transformer for Domain Generalization in Skin Lesion Recognition}, 
      author={Siyuan Yan and Chi Liu and Zhen Yu and Lie Ju and Dwarikanath Mahapatrainst and Victoria Mar and Monika Janda and Peter Soyer and Zongyuan Ge},
      year={2023},
      eprint={2304.01508},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowlegdement

This code is built on [DomainBed](https://github.com/facebookresearch/DomainBed), [DoPrompt](https://github.com/zhengzangw/DoPrompt), and [DG_SKIN](https://github.com/alceubissoto/artifact-generalization-skin) . We thank the authors for sharing their codes.
