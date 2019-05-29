# PaperRobot: Incremental Draft Generation of Scientific Ideas

[Arxiv](https://arxiv.org/pdf/1905.07870.pdf)


Accpeted by 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019)


Table of Contents
=================
  * [Overview](#overview)
  * [Requirements](#requirements)
  * [Quickstart](#quickstart)
  * [Citation](#citation)

## Overview
<p align="center">
  <img src="https://eaglew.github.io/images/paperrobot.png?raw=true" alt="Photo" style="width="100%;"/>
</p>

## Requirements

#### Environment:

- [Pytorch 1.1](http://pytorch.org/)
-  Python 3.6 **CAUTION!! Model might not be saved and loaded properly under Python 3.5**

#### Data:

- [Pubmed term, abstract, conclusion, title dataset](https://drive.google.com/open?id=1O91gX2maPHdIRUb9DdZmUOI5issRMXMY)

## Quickstart

### New paper writing

#### Preprocessing:
Download and unzip the `data_pubmed_writing.zip`. Put `data` folder under the New paper writing folder.

#### Training

Put the type of data after the `--data_path`.  For example, if you want to train an abstract model, put `data/pubmed_abstract` after `--data_path`.
Put the model directory after the `--model_dp`
For more other options, please check the code.
```
python train.py --data_path data/pubmed_abstract --model_dp abstract_model/
```

#### Test
Put the finished model path after the `--model`
The `test.py` will provide the score for test set.
```
python test.py --data_path data/pubmed_abstract --model abstract_model/memory/best_dev_model.pth.tar
```

#### Predict an instance
Put the finished model path after the `--model`
The `input.py` will provide the prediction for customized input.
```
python input.py --data_path data/pubmed_abstract --model abstract_model/memory/best_dev_model.pth.tar
```
## Citation
```
@InProceedings{wang2019paperrobot,
  author = 	"Wang, Qingyun and Huang, Lifu and Jiang, Zhiying and Knight, Kevin and Ji, Heng and Bansal, Mohit and Luan, Yi",
  title = 	"PaperRobot: Incremental Draft Generation of Scientific Ideas",
  booktitle = 	"Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
  year = 	"2019"
}
```
