Data and re-implemented scripts for 2017 AAAI paper, "Robsut Wrod Reocginiton via semi-Character Recurrent Neural Network"

Last updated: Feb 10th , 2017

- - -

This repository contains the re-implementation on the following paper:

    @InProceedings{sakaguchi-duh-post-vandurme:2017:AAAI2017,
      author = {{Sakaguchi}, Keisuke and {Duh}, Kevin and {Post}, Matt and {Van Durme}, Ben},
      title = "{Robsut Wrod Reocginiton via semi-Character Recurrent Neural Network}",
      booktitle = {Proceedings of the Thirty-First {AAAI} Conference on Artificial Intelligence},
      month     = {February},
      year      = {2017},
      address   = {San Francisco, California},
      publisher = {{AAAI} Press},
      pages     = {x--x},
      url       = {https://}
    }


## Data

    .
    ├── README.md  # this file
    ├── train.py # script for training
    ├── predict.py # predict correct word
    ├── models # store model files
    └── binarize.py # utility function

## Basic Usage

    (training) THEANO_FLAGS=device=gpu1,floatX=float32 python train.py
    (predicting) THEANO_FLAGS=device=gpu1,floatX=float32 python predict.py -m models/train_j-INT_n-JUMBLE_u-650_batch-20_ep-10_model.h5


## Questions?
 - Please e-mail to Keisuke Sakaguchi (keisuke[at]cs.jhu.edu).
 - The original code base (written in Chainer) will also be released later on. (now refactoring for the latest version of Chainer)

