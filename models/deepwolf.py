# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import riaan_preprocessing as prep


train, valid = prep.preprocess()

X_train = train[:,:-1]
Y_train = train[:,-1]
X_test = valid[:,:-1]
Y_test = valid[:,-1]


input_dim = len(X_train[1,:])

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, input_dim=input_dim, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(10, init='uniform'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['loss'])

model.fit(X_train, Y_train,
          nb_epoch=20,
          batch_size=16)
score = model.evaluate(X_test, Y_test, batch_size=16)