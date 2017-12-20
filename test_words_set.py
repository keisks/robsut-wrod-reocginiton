"""This module provides the possibility to test a set randoms words"""
import json
import numpy as np
from keras.models import load_model
import binarize

# 1 pre process steps

f = open('vocabID.json','r')
data = json.loads(f.read())
f.close()
vocab = data['voc']
id2vocab = {int(k):v for k,v in data['idvoc'].items()}

def decode_word(X, calc_argmax):
    if calc_argmax:
        X = X.argmax(axis=-1)
    return ' '.join(id2vocab[x] for x in X)


def vectorize_data(vec_cleaned, alph, noise_type = 'JUMBLE', jumble_type='INT'):
    X_vec = np.zeros((int(len(vec_cleaned)),  len(alph) * 3), dtype=np.bool)
    for i, word in enumerate(vec_cleaned):
        if jumble_type == 'NO':
            x_feat, _ = binarize.noise_char(word, noise_type, alph)
        else:
            x_feat, _ = binarize.jumble_char(word, jumble_type, alph)
        X_vec[i] = x_feat

    return X_vec.reshape((1,20,228))

alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:;'*!?`$%&(){}[]-/\@_#"

# 2 test_set and featurization; you can provide a set of 20 words and only 20 words

test_set = ['Otiher', 'brokeryage','fidrms','inclfuding','Merrfill','Lfynch', 'potfential', 'stratfegies', 'mafrket',
            'Frifday', 'afterfnoon', 'invefstment', 'editfions','newsfpapers', 'negowtiated',
            'arranhgement','Ciable', 'Networok', 'undler', 'agrtee']
X_test = vectorize_data(test_set, alph)

# 3 the predictions

model_file = 'models/train_j-INT_n-JUMBLE_u-650_batch-20_ep-10_model.h5'
model = load_model(model_file)
preds = model.predict_classes(X_test, verbose=0)
pred_j = decode_word(preds[0], calc_argmax=False)


print(('WORD', 'SUGGESTION'))
print('-------- + -----------')
for p in zip(test_set, pred_j.split()):
    print(p)


