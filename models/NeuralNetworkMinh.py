from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, merge
from keras.optimizers import SGD, Adam
from keras.layers.core import Dropout
import keras as K
from make_prediction_file import make_prediction_file

import numpy as np
import scipy as sp
import time

def add_residual(x, n_nodes=64):
    y = Dense(n_nodes, activation='relu')(x)
    z = Dense(n_nodes, activation='linear')(y)
    z = merge([x, y], mode='concat')
    z = Activation('relu')(z)
    return z

def resnet():
    x = Input(shape=(44,), name='x')

    res1 = add_residual(x)
    res2 = add_residual(res1)

    main_loss = Dense(1, activation='sigmoid', name='main_output')(res2)
    model = Model(input=x, output=main_loss)
    
    model.compile(optimizer=Adam(), 
              loss='binary_crossentropy', 
              metrics= ['accuracy'])
    return model

data = np.load('../preprocessed/preprocess.npz')

X_train = data['X_train'].astype('float32')
T_train = data['T_train'].astype('int32')
X_valid = data['X_valid'].astype('float32')
T_valid = data['T_valid'].astype('int32')
X_test  = data['X_test'].astype('float32')

positive = np.arange(len(T_train))[T_train == 1]

#act = K.layers.advanced_activations.PReLU(init='zero', weights=None)
#model = Sequential()
print 'FEATURES: %d' % X_train.shape[1]
#model.add(Dense(64, input_dim=X_train.shape[1], init='glorot_uniform'))
#model.add(act)
#model.add(Dense(1)) 
#model.add(Activation("sigmoid"))

model = resnet()

# Make the selection mask the negative ids
negative_ids = np.arange(len(T_train))[T_train == 0]
# selection_mask = (T_train == 0)

p = None
for epoch in range(20):
    negative_samples = np.asarray(np.random.choice( negative_ids, 3 * len(positive), p=p))
    selection = np.concatenate([positive, negative_samples])

    X_train_batch = X_train[selection]
    T_train_batch = T_train[selection]


    model.fit(X_train_batch, T_train_batch, nb_epoch=1, batch_size=32, validation_data=(X_valid, T_valid), verbose = 1)

    p = model.predict(X_train[negative_ids], batch_size=32, verbose = 1).flatten()
    p = p / p.sum()



# classes = model.predict_classes(X_valid, batch_size=32)
proba = model.predict(X_valid, batch_size=32)

def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll


print '\n', np.isnan(proba.flatten()).sum()

print 'Validation log:', logloss(T_valid, proba.flatten().astype('float64'))

make_prediction_file(model.predict(X_test, batch_size=32).flatten().astype('float64'))

# print 'Training error', logloss(T_valid, proba)
