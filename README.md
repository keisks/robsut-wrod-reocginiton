Sample data and simplified scripts used in 2017 AAAI paper, "Robsut Wrod Reocginiton via semi-Character Recurrent Neural Network"

- Keisuke Sakaguchi

Last updated: Dec 27th , 2016

- - -

This repository contains data and scripts in the following paper:

    @ARTICLE{2016arXiv160802214S,
       author = {{Sakaguchi}, K. and {Duh}, K. and {Post}, M. and {Van Durme}, B.
      },
        title = "{Robsut Wrod Reocginiton via semi-Character Recurrent Neural Network}",
      journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
       eprint = {1608.02214},
     primaryClass = "cs.CL",
     keywords = {Computer Science - Computation and Language},
         year = 2016,
        month = aug,
       adsurl = {http://adsabs.harvard.edu/abs/2016arXiv160802214S}
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

## Questions
 - Please e-mail to Keisuke Sakaguchi (keisuke[at]cs.jhu.edu).

