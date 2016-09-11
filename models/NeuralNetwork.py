from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, Adam

from make_prediction_file import make_prediction_file

import numpy as np
import scipy as sp
import time

data = np.load('../preprocessed/preprocess.npz')

X_train = data['X_train'].astype('float32')
T_train = data['T_train'].astype('int32')
X_valid = data['X_valid'].astype('float32')
T_valid = data['T_valid'].astype('int32')
X_test  = data['X_test'].astype('float32')

positive = np.arange(len(T_train))[T_train == 1]

model = Sequential()
print 'FEATURES: %d' % X_train.shape[1]
model.add(Dense(64, input_dim=X_train.shape[1], init='glorot_uniform'))
model.add(Activation("relu"))
model.add(Dense(1)) 
model.add(Activation("sigmoid"))

sgd = Adam(lr=0.001)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Make the selection mask the negative ids
negative_ids = np.arange(len(T_train))[T_train == 0]
# selection_mask = (T_train == 0)

p = None
for epoch in range(20):
    negative_samples = np.asarray(np.random.choice( negative_ids, 3 * len(positive), p=p))
    selection = np.concatenate([positive, negative_samples])

    X_train_batch = X_train[selection]
    T_train_batch = T_train[selection]


    model.fit(X_train_batch, T_train_batch, nb_epoch=1, batch_size=32, validation_data=(X_valid, T_valid))

    p = model.predict_proba(X_train[negative_ids], batch_size=32).flatten()
    p = p / p.sum()



# classes = model.predict_classes(X_valid, batch_size=32)
proba = model.predict_proba(X_valid, batch_size=32)

def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll


print '\n', np.isnan(proba.flatten()).sum()

print 'Validation log:', logloss(T_valid, proba.flatten().astype('float64'))

make_prediction_file(model.predict_proba(X_test, batch_size=32).flatten().astype('float64'))

# print 'Training error', logloss(T_valid, proba)
