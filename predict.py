#!/usr/bin/env python
from __future__ import print_function # compatible print function for py2 print()
import argparse
import math
import os
import sys
import time
import datetime
import random
import numpy as np
import six
import csv

import binarize
import h5py
from keras.models import Sequential, load_model
from keras.layers import Activation, Dense, Dropout, LSTM, TimeDistributed
from keras.optimizers import SGD

parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', '-b', type=int, default=20,
    help='learning minibatch size')
parser.add_argument('--noise', '-n', default="JUMBLE",
    help='noise type (JUMBLE, INSERT, DELETE, REPLACE, RANDOM)')
parser.add_argument('--jumble', '-j', default="INT",
    help='jumble position (INT, WHOLE, BEG, or END)')
parser.add_argument('--model', '-m', help='model file path')

### path for data (for example)
PATH_TRAIN = './data/ptb.train.txt'
PATH_DEV = './data/ptb.valid.txt'
PATH_TEST = './data/ptb.test.txt'

args = parser.parse_args()
batchsize = args.batchsize  # minibatch size
noise_type = args.noise     # noise type 
jumble_type = args.jumble   # jumble position
model_file = args.model   # jumble position
assert noise_type in ['JUMBLE', 'INSERT', 'DELETE', 'REPLACE', 'RANDOM']
assert jumble_type in ['INT', 'WHOLE', 'BEG', 'END']
assert os.path.exists(model_file)
if not noise_type in ['JUMBLE', 'RANDOM']:
    jumble_type = "NO"

print("===== LOADING VOCAB =====")
vocab = {}
id2vocab = {}

def colors(token, color='green'):
   c_green = '\033[92m' # green
   c_red = '\033[91m' # red
   c_close = '\033[0m' # close
   return c_green + token + c_close

def load_data(filename):
    global vocab

    words = open(filename).read().replace('\n', '<eos>').strip().split()
    dataset = np.ndarray((len(words),), dtype=np.int32)
    for i, word in enumerate(words):
        if word not in vocab:
            # put one hot vector: len(vocab) as a index
            vocab[word] = len(vocab) 
            id2vocab[vocab[word]] = word
            # present input data as a sequence of one-hot vector
        dataset[i] = vocab[word]
    return dataset

def decode_word(X, calc_argmax):
    if calc_argmax:
        X = X.argmax(axis=-1)
    return ' '.join(id2vocab[x] for x in X)

alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:;'*!?`$%&(){}[]-/\@_#" 
# NB. # is <eos>, _ is <unk>, @ is number

# sentence is represented as id, <eos> is also represented as one word

train_data = load_data(PATH_TRAIN)
dev_data = load_data(PATH_DEV)
test_data = load_data(PATH_TEST)
test_cleaned = open('./data/ptb.test.txt').read().replace('\n', '<eos>').strip().split()

print('#vocab:\t', len(vocab)-2) # excluding BOS, EOS
print('#tokens in test:\t', len(test_cleaned))


print("===== VECTORIZING DATA =====")
timesteps = len(test_cleaned)
data_dim = len(alph)*3

def vectorize_data(vec_cleaned, data_name): # training, dev, or test
    X_vec = np.zeros((int(len(vec_cleaned)/batchsize), batchsize, len(alph)*3), dtype=np.bool)
    Y_vec = np.zeros((int(len(vec_cleaned)/batchsize), batchsize, len(vocab)), dtype=np.bool)
    X_token = []
    # easy minibatch
    # https://docs.python.org/2.7/library/functions.html?highlight=zip#zip
    for m, mini_batch_tokens in enumerate(zip(*[iter(vec_cleaned)]*batchsize)):
        X_token_m = []
        x_mini_batch = np.zeros((batchsize, len(alph)*3), dtype=np.bool)
        y_mini_batch = np.zeros((batchsize, len(vocab)), dtype=np.bool)
     
        for j, token in enumerate(mini_batch_tokens):
            if jumble_type == 'NO':
                x_mini_batch[j], x_token = binarize.noise_char(token, noise_type, alph)
                pass
            else:
                x_mini_batch[j], x_token = binarize.jumble_char(token, jumble_type, alph)

            bin_label = [0]*len(vocab)
            bin_label[vocab[token]] = 1
            y_mini_batch[j] = np.array(bin_label)
            X_token_m.append(x_token)
        X_vec[m] = x_mini_batch
        Y_vec[m] = y_mini_batch
        X_token.append(X_token_m)

        #percentage = int(m*100. / (len(vec_cleaned)/batchsize))
        #sys.stdout.write("\r%d %% %s" % (percentage, data_name))
        #print(str(percentage) + '%'),
        #sys.stdout.flush()
    print()
    return X_vec, Y_vec, X_token


X_test, Y_test, X_test_token = vectorize_data(test_cleaned, 'for test data')

print("data shape (#_batches, batch_size, vector_size)")
print("X_test", X_test.shape)
print("Y_test", Y_test.shape)


#LOAD the model 
model = load_model(model_file)

for j in range(len(X_test)):
    x_raw, y_raw = X_test[np.array([j])], Y_test[np.array([j])]
    src_j = " ".join(X_test_token[j])
    ref_j = decode_word(y_raw[0], calc_argmax=True)
    preds = model.predict_classes(x_raw, verbose=0)
    pred_j = decode_word(preds[0], calc_argmax=False)

    # coloring
    pred_j_list = pred_j.split()
    ref_j_list = ref_j.split()
    for k in range(len(pred_j_list)):
        if pred_j_list[k] == ref_j_list[k]:
            pred_j_list[k] = colors(pred_j_list[k])
    pred_j = " ".join(pred_j_list)

    print('example #', str(j+1))
    print('src: ', src_j)
    print('prd: ', pred_j)
    print('ref: ', ref_j)

