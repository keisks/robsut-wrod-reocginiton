#!/usr/bin/python
from __future__ import print_function # compatible print function for py2 print()
import argparse
import math
import sys
import time
import datetime
import random
import numpy as np
import six
import csv

import binarize
import h5py
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM, TimeDistributed
from keras.optimizers import SGD

### path for data (for example)
PATH_TRAIN = './data/ptb.train.txt'
PATH_DEV = './data/ptb.valid.txt'
PATH_TEST = './data/ptb.test.txt'

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', '-e', default=30, type=int,
    help='number of epochs to learn')
parser.add_argument('--unit', '-u', default=650, type=int,
    help='number of units in hidden layers')
parser.add_argument('--batchsize', '-b', type=int, default=20,
    help='learning minibatch size')
parser.add_argument('--checkpoint', '-c', type=int, default=5,
    help='checkpoint for saving the model (# of epoch)')
parser.add_argument('--noise', '-n', default="JUMBLE",
    help='noise type (JUMBLE, INSERT, DELETE, REPLACE, RANDOM)')
parser.add_argument('--jumble', '-j', default="INT",
    help='jumble position (INT, WHOLE, BEG, or END)')
parser.add_argument('--pilot', '-p', default=False, action='store_true',
    help='If True, results and model are not saved (Default: False)')


args = parser.parse_args()

n_epoch = args.epoch        # number of epochs
n_units = args.unit         # number of units per layer
batchsize = args.batchsize  # minibatch size
check_point = args.checkpoint # checkpoint (num epoch)
noise_type = args.noise     # noise type 
is_pilot = args.pilot
assert noise_type in ['JUMBLE', 'INSERT', 'DELETE', 'REPLACE', 'RANDOM']
jumble_type = args.jumble   # jumble position
assert jumble_type in ['INT', 'WHOLE', 'BEG', 'END']
if not noise_type in ['JUMBLE', 'RANDOM']:
    jumble_type = "NO"

print("===== EXP SETTING =====")
print("num epoch:\t"   +str(n_epoch))
print("num units:\t"   +str(n_units))
print("batch size:\t"  +str(batchsize))
print("noise type:\t"  +noise_type)
print("jumble type:\t" +jumble_type)
print("is pilot?:\t"   +str(is_pilot))

EXP_NAME = "train_j-"+ jumble_type + "_n-" + noise_type + "_u-" + str(n_units) + '_batch-' + str(batchsize)

d = datetime.datetime.today()
START_TIME =  d.strftime('%Y/%m/%d %H:%M:%S')

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

    # words are considered in a document level
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

train_data = ""
train_cleaned = ""
if is_pilot:
    train_data = load_data(PATH_TRAIN)
    train_cleaned = open(PATH_TRAIN).read().replace('\n', '<eos>').strip().split()
else:
    train_data = load_data(PATH_TRAIN)
    train_cleaned = open(PATH_TRAIN).read().replace('\n', '<eos>').strip().split()

dev_data = load_data(PATH_DEV)
test_data = load_data(PATH_TEST)
dev_cleaned = open(PATH_DEV).read().replace('\n', '<eos>').strip().split()

print('#vocab:\t', len(vocab)-2) # excluding BOS, EOS
print('#tokens in training:\t', len(train_cleaned))
print('#tokens in validation:\t', len(dev_cleaned))


print("===== VECTORIZING DATA =====")
timesteps = len(train_cleaned)
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
            else:
                x_mini_batch[j], x_token = binarize.jumble_char(token, jumble_type, alph)

            bin_label = [0]*len(vocab)
            bin_label[vocab[token]] = 1
            y_mini_batch[j] = np.array(bin_label)
            X_token_m.append(x_token)
        X_vec[m] = x_mini_batch
        Y_vec[m] = y_mini_batch
        X_token.append(X_token_m)

        percentage = int(m*100. / (len(vec_cleaned)/batchsize))
        sys.stdout.write("\r%d %% %s" % (percentage, data_name))
        #print(str(percentage) + '%'),
        sys.stdout.flush()
    print()
    return X_vec, Y_vec, X_token


X_train, Y_train, X_train_token = vectorize_data(train_cleaned, 'for train data')
X_dev, Y_dev, X_dev_token = vectorize_data(dev_cleaned, 'for dev data')

print("data shape (#_batches, batch_size, vector_size)")
print("X_train", X_train.shape)
print("Y_train", Y_train.shape)
print("X_dev", X_dev.shape)
print("Y_dev", Y_dev.shape)


model = Sequential()

model.add(LSTM(n_units, return_sequences=True, batch_input_shape=(None, batchsize, data_dim)))
model.add(Dropout(0.01))
model.add(TimeDistributed(Dense(len(vocab))))
model.add(Activation('softmax'))
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
        optimizer='rmsprop', # or sgd
        #optimizer='sgd', # or sgd
        metrics=['accuracy'])


if not is_pilot:
    result_file = open('./results/' + EXP_NAME +'.result', 'w')
    result_csv = csv.writer(result_file)
    result_csv.writerow(['epoch', 'loss', 'acc', 'val_acc', 'val_loss'])


print("===== TRAINING START =====")
for epoch_i in range(1, n_epoch+1):
    print("--- Epoch " + str(epoch_i) + " -----")
    hist = model.fit(X_train, Y_train, nb_epoch=1, validation_data=(X_dev, Y_dev))
    hist = hist.history
    #print(hist.history)
    # e.g. hist = {
    #   'loss': [6.6267016227313018], 
    #   'acc': [0.1131309146525732], 
    #   'val_acc': [0.21434720261053627], 
    #   'val_loss': [6.302218198434181]}
    if not is_pilot:
        result_csv.writerow([str(epoch_i), hist['loss'][0], hist['acc'][0], hist['val_acc'][0], hist['val_loss'][0]])

    if epoch_i % check_point == 0: # check point
        # save the model
        if not is_pilot:
            #model.save_weights('./models/' + EXP_NAME + '_ep-' +str(epoch_i) + '_weights.h5')
            model.save('./models/' + EXP_NAME + '_ep-' +str(epoch_i) + '_model.h5')

        # check output
        for j in range(5):
            x_raw, y_raw = X_dev[np.array([j])], Y_dev[np.array([j])]
            src_j = " ".join(X_dev_token[j])
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

print("===== TRAINING FINISHED =====")

if not is_pilot:
    result_file.close()

d = datetime.datetime.today()
END_TIME =  d.strftime('%Y/%m/%d %H:%M:%S')

