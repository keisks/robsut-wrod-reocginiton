Data and re-implemented scripts for 2017 AAAI paper, "Robsut Wrod Reocginiton via semi-Character Recurrent Neural Network"

Last updated: Oct 8th , 2017

- - -

This repository contains the re-implementation on the following paper:

    @inproceedings{DBLP:conf/aaai/SakaguchiDPD17,
      author    = {Keisuke Sakaguchi and
                   Kevin Duh and
                   Matt Post and
                   Benjamin Van Durme},
      title     = {Robsut Wrod Reocginiton via Semi-Character Recurrent Neural Network},
      booktitle = {Proceedings of the Thirty-First {AAAI} Conference on Artificial Intelligence,
                   February 4-9, 2017, San Francisco, California, {USA.}},
      pages     = {3281--3287},
      year      = {2017},
      url       = {http://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14332},
      publisher = {{AAAI} Press},
    }
        

## Data

    .
    ├── README.md  # this file
    ├── train.py # script for training
    ├── predict.py # predict correct word
    ├── models # store model files
    └── binarize.py # utility function

## Basic Usage

    (pre-requisite)

        conda create -n robsut python=2.7
        pip install keras theano h5py
        change the keras backend to theano (edit $HOME/.keras/keras.json)

    (training) 

        THEANO_FLAGS=device=gpu0,floatX=float32 python train.py

    (predicting) 

        THEANO_FLAGS=device=gpu0,floatX=float32 python predict.py -m models/train_j-INT_n-JUMBLE_u-650_batch-20_ep-10_model.h5


## Questions?
 - Please e-mail to Keisuke Sakaguchi (keisuke[at]cs.jhu.edu).

