# EPVT: Environment-aware Prompt Vision Transformer for Domain Generalization in Skin Lesion Recognition<br><sub><sub>Official PyTorch implementation of the MICCAI 2023 paper</sub></sub>
[[`arXiv`](https://arxiv.org/pdf/2304.01508.pdf)]
[[`BibTex`](#citation)]
[[`MICCAI paper`](https://link.springer.com/chapter/10.1007/978-3-031-43990-2_24)]

## Introduction
[**EPVT: Environment-aware Prompt Vision Transformer for Domain Generalization in Skin Lesion Recognition**](https://arxiv.org/pdf/2304.01508.pdf)<br>
Siyuan Yan, Chi Liu, Zhen Yu, Lie Ju, Dwarikanath Mahapatra, Victoria Mar, Monika Janda, Peter Soyer, Zongyuan Ge


**[abstract]** *Skin lesion recognition using deep learning has made remarkable progress, and there is an increasing need for deploying these systems in real-world scenarios. However, recent research has revealed that deep neural networks for skin lesion recognition may overly depend on disease-irrelevant image artifacts (i.e. dark corners, dense hairs), leading to poor generalization in unseen environments. To address this issue, we propose a novel domain generalization method called EPVT, which involves embedding prompts into the vision transformer to collaboratively learn knowledge from diverse domains. Concretely, EPVT leverages a set of domain prompts, each of which plays as a domain expert, to capture domain-specific knowledge; and a shared prompt for general knowledge over the entire dataset. To facilitate knowledge sharing and the interaction of different prompts, we introduce a domain prompt generator that enables low-rank multiplicative updates between domain prompts and the shared prompt. A domain mixup strategy is additionally devised to reduce the co-occurring artifacts in each domain, which allows for more flexible decision margins and mitigates the issue of incorrectly assigned domain labels. Experiments on four out-of-distribution datasets and six different biased ISIC datasets demonstrate the superior generalization ability of EPVT in skin lesion recognition across various environments.*
![alt text](image/motivation.png)
## Method
<img src="image/method.png" alt="My Image" width="800">

## Training

Our benchmark is modified based on DomainBed, please refer to [DomainBed Readme](README_domainbed.md) for more details on commands running jobs. 

```sh
# OfficeHome ERM
python -m domainbed.scripts.train --data_dir=./domainbed/data/ --steps 5001 --dataset OfficeHome --test_env 0/1/2/3 --algorithm DoPrompt --output_dir results/exp \
     --hparams '{"lr": 1e-5, "lr_classifier": 1e-4}'
# OfficeHome
python -m domainbed.scripts.train --data_dir=./domainbed/data/ --steps 5001 --dataset OfficeHome --test_env 0/1/2/3 --algorithm DoPrompt --output_dir results/exp \
     --hparams '{"lr": 1e-5, "lr_classifier": 1e-3}'
# PACS ERM
python -m domainbed.scripts.train --data_dir=./domainbed/data/ --steps 5001 --dataset PACS --test_env 0/2/3 --algorithm DoPrompt --output_dir results/exp \
     --hparams '{"lr": 5e-6, "lr_classifier": 5e-5}'
# PACS
python -m domainbed.scripts.train --data_dir=./domainbed/data/ --steps 5001 --dataset PACS --test_env 0/2/3 --algorithm DoPrompt --output_dir results/exp \
     --hparams '{"lr": 5e-6, "lr_classifier": 5e-4}'
# VLCS ERM
python -m domainbed.scripts.train --data_dir=./domainbed/data/ --steps 5001 --dataset VLCS --test_env 0/1/2/3 --algorithm DoPrompt --output_dir results/exp \
     --hparams '{"lr": 5e-6, "lr_classifier": 5e-5}'
# VLCS
python -m domainbed.scripts.train --data_dir=./domainbed/data/ --steps 5001 --dataset VLCS --test_env 0/1/2/3 --algorithm DoPrompt --output_dir results/exp \
     --hparams '{"lr": 5e-6, "lr_classifier": 5e-6}'
```

## Collect Results

```sh
python -m domainbed.scripts.collect_results --input_dir=results
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
