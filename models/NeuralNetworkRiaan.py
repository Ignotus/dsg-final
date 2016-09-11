from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, merge
from keras.optimizers import SGD, Adam
from keras.layers.core import Dropout
import keras as K
from make_prediction_file import make_prediction_file

from keras.regularizers import l2, activity_l2

import numpy as np
import scipy as sp
import time

def resnet():
    x = Input(shape=(44,), name='x')
    
    #y = Dense(128, activation='relu', init= 'glorot_normal')(x)
    
    #n = merge([x, y], mode = 'concat')
    
    a = Dense(128, activation='relu', init= 'glorot_normal', W_regularizer=l2(0.001), activity_regularizer=activity_l2(0.001))(x)
    
    z = merge([x, a], mode='concat')
    
    
    main_loss = Dense(1, activation='sigmoid', name='main_output', init= 'glorot_normal', W_regularizer=l2(0.001), activity_regularizer=activity_l2(0.001))(z)
    model = Model(input=x, output=main_loss)
    
    model.compile(optimizer=Adam(), 
              loss='binary_crossentropy', 
              metrics= ['accuracy'])
    return model

data = np.load('../preprocessed/preprocess.npz')

X_train = data['X_train_all'].astype('float32')
T_train = data['T_train_all'].astype('int32')
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
model.summary()
# Make the selection mask the negative ids
negative_ids = np.arange(len(T_train))[T_train == 0]
# selection_mask = (T_train == 0)

p = None
for epoch in range(10):
    negative_samples = np.asarray(np.random.choice( negative_ids, 3 * len(positive), p=p))
    selection = np.concatenate([positive, negative_samples])

    X_train_batch = X_train[selection]
    T_train_batch = T_train[selection]


    model.fit(X_train_batch, T_train_batch, nb_epoch=1, batch_size=64, validation_data=(X_valid, T_valid), verbose = 1)

    p = model.predict(X_train[negative_ids], batch_size=32, verbose = 0).flatten()
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

model.save_weights('weights_resnet_2l_44i_64.h5')
make_prediction_file(model.predict(X_test, batch_size=32, verbose = 0).flatten().astype('float64'))

# print 'Training error', logloss(T_valid, proba)
