# PaperRobot: Incremental Draft Generation of Scientific Ideas

[[Arxiv]](https://arxiv.org/pdf/1905.07870.pdf) [[Sample Output]](https://eaglew.github.io/PaperRobot/)


Accepted by 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019)


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

- Python 3.6 **CAUTION!! Model might not be saved and loaded properly under Python 3.5**
- Ubuntu 16.04/18.04 **CAUTION!! Model might not run properly on windows because [windows uses backslashes on path while linux/OS X uses forward slashes](https://www.howtogeek.com/181774/why-windows-uses-backslashes-and-everything-else-uses-forward-slashes/)**
- [Pytorch 1.1](http://pytorch.org/)
- [NumPy 1.16.3](https://www.scipy.org/install.html)
- [SciPy 1.2.1](https://www.scipy.org/install.html)
- [NetworkX 2.3](https://networkx.github.io/documentation/stable/install.html)

#### Data:

- [PubMed Paper Reading  Dataset](https://drive.google.com/open?id=1DLmxK5x7m8bDPK5ZfAROtGpkWZ_v980Z)
This dataset gathers 14,857 entities, 133 relations, and entities corresponding tokenized text from PubMed. It contains 875,698 training pairs, 109,462 development pairs, and 109,462 test pairs.

- [PubMed Term, Abstract, Conclusion, Title Dataset](https://drive.google.com/open?id=1O91gX2maPHdIRUb9DdZmUOI5issRMXMY)
This dataset gathers three types of pairs: Title-to-Abstract (Training: 22,811/Development: 2095/Test: 2095), Abstract-to-Conclusion and Future work (Training: 22,811/Development: 2095/Test: 2095), Couclusion and Future work-to-Title (Training: 15,902/Development: 2095/Test: 2095) from PubMed. Each pair contains a pair of input and output as well as the corresponding terms(from original KB and link prediction results).

## Quickstart

### Existing paper reading

**CAUTION!! Because the dataset is quite large, the training and evaluation of link prediction model will be pretty slow.**

#### Preprocessing:
Download and unzip the `paper_reading.zip` from [PubMed Paper Reading  Dataset](https://drive.google.com/open?id=1DLmxK5x7m8bDPK5ZfAROtGpkWZ_v980Z)
. Put `paper_reading` folder under the `Existing paper reading` folder.

#### Training

Hyperparameter can be adjusted as follows: For example, if you want to change the number of hidden unit to 6, you can append `--hidden 6` after `train.py`
```
python train.py
```


#### Test

Put the finished model path after the `--model`
The `test.py` will provide the ranking score for the test set.
```
python test.py --model models/GATA/best_dev_model.pth.tar
```

### New paper writing

#### Preprocessing:

Download and unzip the `data_pubmed_writing.zip` from [PubMed Term, Abstract, Conclusion, Title Dataset](https://drive.google.com/open?id=1O91gX2maPHdIRUb9DdZmUOI5issRMXMY)
. Put `data` folder under the `New paper writing folder`.

#### Training

Put the type of data after the `--data_path`.  For example, if you want to train an abstract model, put `data/pubmed_abstract` after `--data_path`.
Put the model directory after the `--model_dp`
For more other options, please check the code.
```
python train.py --data_path data/pubmed_abstract --model_dp abstract_model/
```

#### Test
Put the finished model path after the `--model`
The `test.py` will provide the score for the test set.
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
@inproceedings{wang-etal-2019-paperrobot,
    title = "{P}aper{R}obot: Incremental Draft Generation of Scientific Ideas",
    author = "Wang, Qingyun  and
      Huang, Lifu  and
      Jiang, Zhiying  and
      Knight, Kevin  and
      Ji, Heng  and
      Bansal, Mohit  and
      Luan, Yi",
    booktitle = "Proceedings of the 57th Conference of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1191",
    pages = "1980--1991"
}
```
