#!/usr/bin/env python
#encoding: utf-8

import string
import numpy as np
import random

def hasnum(w):
    for c_i in w:
        if c_i.isdigit():
            return True
    return False

def noise_char(w, opt, alph):
    if opt == "DELETE":
        bin_all = [0]*len(alph)
        if w == '<eos>':
            bin_all[-1] += 1
        elif w == '<unk>':
            bin_all[-2] += 1
        elif hasnum(w):
            bin_all[-3] += 1
        else:
            idx = random.randint(0, len(w)-1)
            w = w[:idx] + w[idx+1:]
            for i in range(len(w)):
                try:
                    bin_all[alph.index(w[i])] += 1
                except ValueError:
                    print(w)
                    raise

        return np.array(bin_all*3), w

    if opt == "INSERT":
        bin_all = [0]*len(alph)
        if w == '<eos>':
            bin_all[-1] += 1
        elif w == '<unk>':
            bin_all[-2] += 1
        elif hasnum(w):
            bin_all[-3] += 1
        else:
            ins_idx = random.randint(0, len(w)-1)
            ins_char_idx = np.random.randint(0,len(string.ascii_lowercase))
            ins_char = list(string.ascii_lowercase)[ins_char_idx]
            w = w[:ins_idx] + ins_char + w[ins_idx:]
            for i in range(len(w)):
                try:
                    bin_all[alph.index(w[i])] += 1
                except ValueError:
                    print(w)
                    raise

        return np.array(bin_all*3), w

    if opt == "REPLACE":
        bin_all = [0]*len(alph)
        if w == '<eos>':
            bin_all[-1] += 1
        elif w == '<unk>':
            bin_all[-2] += 1
        elif hasnum(w):
            bin_all[-3] += 1
        else:
            target_idx = random.randint(0, len(w)-1)
            rep_char_idx = np.random.randint(0,len(string.ascii_lowercase))
            rep_char = list(string.ascii_lowercase)[rep_char_idx]
            w = w[:target_idx] + rep_char + w[target_idx+1:]
            for i in range(len(w)):
                try:
                    bin_all[alph.index(w[i])] += 1
                except ValueError:
                    print(w)
                    raise

        return np.array(bin_all*3), w


def jumble_char(w, opt, alph):
    if opt == "WHOLE":
        bin_all = [0]*len(alph)
        bin_filler = [0]*(len(alph)*2)
        if w == '<eos>':
            bin_all[-1] += 1
        elif w == '<unk>':
            bin_all[-2] += 1
        elif hasnum(w):
            bin_all[-3] += 1
        else:
            w = ''.join(random.sample(w, len(w)))
            for i in range(len(w)):
                try:
                    bin_all[alph.index(w[i])] += 1
                except ValueError:
                    print(w)
                    raise

        bin_all = bin_all + bin_filler
        return np.array(bin_all), w

    elif opt == "BEG":
        bin_initial = [0]*len(alph)
        bin_end = [0]*len(alph)
        bin_filler = [0]*len(alph)
        if w == '<eos>':
            bin_initial[-1] += 1
            bin_end[-1] += 1
        elif w == '<unk>':
            bin_initial[-2] += 1
            bin_end[-2] += 1
        elif hasnum(w):
            bin_initial[-3] += 1
            bin_end[-3] += 1
        else:
            if len(w) > 3:
                w_init = ''.join(random.sample(w[:-1], len(w[:-1])))
                w = w_init + w[-1]
            for i in range(len(w)):
                try:
                    if i==len(w)-1:
                        bin_end[alph.index(w[i])] += 1
                    else:
                        bin_initial[alph.index(w[i])] += 1
                except ValueError:
                    print(w)
                    raise
        bin_all = bin_initial + bin_end + bin_filler
        return np.array(bin_all), w

    elif opt == "END":
        bin_initial = [0]*len(alph)
        bin_end = [0]*len(alph)
        bin_filler = [0]*len(alph)
        if w == '<eos>':
            bin_initial[-1] += 1
            bin_end[-1] += 1
        elif w == '<unk>':
            bin_initial[-2] += 1
            bin_end[-2] += 1
        elif hasnum(w):
            bin_initial[-3] += 1
            bin_end[-3] += 1
        else:
            if len(w) > 3:
                w_end = ''.join(random.sample(w[1:], len(w[1:])))
                w = w[0] + w_end
            for i in range(len(w)):
                try:
                    if i==0:
                        bin_initial[alph.index(w[i])] += 1
                    else:
                        bin_end[alph.index(w[i])] += 1
                except ValueError:
                    print(w)
                    raise
        bin_all = bin_initial + bin_end + bin_filler
        return np.array(bin_all), w
    
    elif opt == "INT":
        bin_initial = [0]*len(alph)
        bin_middle = [0]*len(alph)
        bin_end = [0]*len(alph)
        if w == '<eos>':
            bin_initial[-1] += 1
            bin_middle[-1] += 1
            bin_end[-1] += 1
        elif w == '<unk>':
            bin_initial[-2] += 1
            bin_middle[-2] += 1
            bin_end[-2] += 1
        elif hasnum(w):
            bin_initial[-3] += 1
            bin_middle[-3] += 1
            bin_end[-3] += 1
        else:
            if len(w) > 3:
                w_mid = ''.join(random.sample(w[1:-1], len(w[1:-1])))
                w = w[0] + w_mid + w[-1]
            for i in range(len(w)):
                try:
                    if i==0:
                        bin_initial[alph.index(w[i])] += 1
                    elif i==len(w)-1:
                        bin_end[alph.index(w[i])] += 1
                    else:
                        bin_middle[alph.index(w[i])] += 1
                except ValueError:
                    print(w)
                    raise
        bin_all = bin_initial + bin_middle + bin_end
        return np.array(bin_all), w
    else:
        raise


if __name__ == "__main__":
    alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:;'*!?`$%&(){}[]-/\@_#" 
    word = 'research'
    print(word)
    v, w = noise_char(word, 'DELETE', alph)
    print(w)
    v, w = noise_char(word, 'INSERT', alph)
    print(w)
    v, w = noise_char(word, 'REPLACE', alph)
    print(w)


