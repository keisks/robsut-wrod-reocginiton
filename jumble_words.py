#!/usr/bin/env python
#encoding: utf-8

import os, sys
import random
import argparse

def jumble(w, jumble_type):
    if jumble_type == "internal":
        target_chars = list(w[1:-1])
        random.shuffle(target_chars)
        jumbled_word = "".join(list(w[0]) + target_chars + list(w[-1]))
    elif jumble_type == "beginning":
        target_chars = list(w[:-1])
        random.shuffle(target_chars)
        jumbled_word = "".join(target_chars + list(w[-1]))
    elif jumble_type == "ending":
        target_chars = list(w[1:])
        random.shuffle(target_chars)
        jumbled_word = "".join(list(w[0]) + target_chars)
    elif jumble_type == "whole":
        target_chars = list(w)
        random.shuffle(target_chars)
        jumbled_word = "".join(target_chars)
    else: # debug
        raise
    return jumbled_word

def jumble_data(clean_words, jumble_type):
    jumbled =  []
    for w_i in clean_words:
        if len(w_i) <= 3:
            jumbled.append(w_i)
        elif w_i in ['\n', "<unk>"]:
            jumbled.append(w_i)
        else:
            jumbled.append(jumble(w_i, jumble_type))
    return jumbled

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--jumble', dest="jumble_type", choices=["internal", "beginning", "ending", "whole"], required=True,
                                        default="internal", help="jumble type to apply")
    arg_parser.add_argument('--data', dest="data_path", required=True,
                                        default=None, help="data file path")
    args = arg_parser.parse_args()
    for row in open(args.data_path, 'r'):
        print(" " + " ".join(jumble_data(row.split(), args.jumble_type)) + " ")

