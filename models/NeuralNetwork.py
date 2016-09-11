from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam
from keras.models import load_model

from make_prediction_file import make_prediction_file

import numpy as np
import scipy as sp
import time
import os

def logloss(act, pred, epsilon=1e-15):
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

def build_model(X_train):
    model = Sequential()
    print 'FEATURES: %d' % X_train.shape[1]
    model.add(Dense(64, input_dim=X_train.shape[1], init='glorot_uniform'))
    model.add(Activation("tanh"))
    # model.add(Dropout(0.5))
    model.add(Dense(32, init='glorot_uniform'))
    model.add(Activation("tanh"))
    model.add(Dense(1)) 
    model.add(Activation("sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model

def train_model(model, X_train, T_train, X_valid=None, T_valid=None, epochs=20):
    sample_weights = None
    positive_ids = np.arange(len(T_train))[T_train == 1]
    negative_ids = np.arange(len(T_train))[T_train == 0]

    for epoch in range(epochs):
        print '\nEPOCH %d --------------' % epoch

        negative_samples = np.asarray(np.random.choice( negative_ids, 3 * len(positive_ids), p=sample_weights))
        selection = np.concatenate([positive_ids, negative_samples])

        X_train_selection = X_train[selection]
        T_train_selection = T_train[selection]

        if X_valid is not None and T_valid is not None:
            model.fit(X_train_selection, T_train_selection, nb_epoch=1, batch_size=32, validation_data=(X_valid, T_valid))
        else:
            model.fit(X_train_selection, T_train_selection, nb_epoch=1, batch_size=32)

        sample_weights = model.predict_proba(X_train[negative_ids], batch_size=32).flatten() + 0.01 # smoothing 
        sample_weights = sample_weights / sample_weights.sum()


def main(datadir='../preprocessed/', modelfn='model_NN.h5'):
    data = np.load(os.path.join(datadir, 'preprocess.npz'))

    X_train = data['X_train'].astype('float32')
    T_train = data['T_train'].astype('int32')
    X_valid = data['X_valid'].astype('float32')
    T_valid = data['T_valid'].astype('int32')

    # X_train = data['X_train_all'].astype('float32')
    # T_train = data['T_train_all'].astype('int32')
    # X_valid = None #data['X_valid'].astype('float32')
    # T_valid = None # data['T_valid'].astype('int32')

    model = build_model(X_train)
    train_model(model, X_train, T_train, X_valid, T_valid)

    if X_valid is not None:
        print '\nValidation log:', logloss(T_valid, model.predict_proba(X_valid, batch_size=32).flatten().astype('float64'))

    model.save(modelfn)

def predict(datadir='../preprocessed/', modelfn='model_NN.h5'):
    model = load_model(modelfn)

    data = np.load(os.path.join(datadir, 'preprocess.npz'))
    X_test  = data['X_test'].astype('float32')

    make_prediction_file(model.predict_proba(X_test, batch_size=32).flatten().astype('float64'))


if __name__ == '__main__':
    main()

    predict()

# print 'Training error', logloss(T_valid, proba)
