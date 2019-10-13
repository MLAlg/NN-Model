#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 14:07:33 2019

@author: samaneh
"""

import numpy as np
import matplotlib.pyplot as plt

from dataset import load_hoda
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
np.random.seed(123) # for reproducibility

X_train_org, y_train_org, X_test_org, y_test_org = load_hoda()

def print_data(x_train, y_train, x_test, y_test):
    print("\t type(X_train): {}".format(type(x_train)))
    print("\t type(y_train): {}".format(type(y_train)))
    print("\t X_train.shape: {}".format(np.shape(x_train)))
    print("\t y_train.shape: {}".format(np.shape(y_train)))
    print("\t y_train[0]: {}".format(type(y_train[0])))

# preprocess input data for keras    
X_train = np.array(X_train_org) # input data in numpy array format
y_train = keras.utils.to_categorical(y_train_org, num_classes=10) # one-hot encoding
X_test = np.array(X_test_org) 
y_test = keras.utils.to_categorical(y_test_org, num_classes=10) 

print("Before preprocessing: ", print_data(X_train_org, y_train_org, X_test_org, y_test_org))
print("After preprocessing: ", print_data(X_train, y_train, X_test, y_test))

 # Normalize data values to the range [0,1]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# we create a simple model first
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=25)) # input layer
model.add(Dense(64, activation='relu')) # hidden layer
model.add(Dense(200, activation='relu')) # hidden layer
model.add(Dense(10, activation='softmax')) # output layer
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, batch_size=64, validation_split=0.2)

loss, acc = model.evaluate(X_test, y_test)
print('\n Test loss: %.2f, \t accuracy: %.2f %%' %(loss, acc))
pred_class = model.predict_classes(X_test)
print("predicted: ", pred_class, "\n true class: ", y_test_org)

# improve the architecture of the model
#######################################
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=25)) # input layer
model.add(Dense(200, activation='relu')) # hidden layer
model.add(Dense(10, activation='softmax')) # output layer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_loss = []
train_acc = []
test_loss = []
test_acc = []
for i in range(200):
    model.fit(X_train, y_train, epochs=1, batch_size=128, verbose=0) # fit model on training data
    loss, acc = model.evaluate(X_train, y_train, verbose=0) # verbose=0 --> do not print anything
    train_loss += [loss]
    train_acc += [acc]
    loss, acc = model.evaluate(X_test, y_test, verbose=0) # verbose=0 --> do not print anything
    test_loss += [loss]
    test_acc += [acc]
    
print('\n Test accuracy: ', test_acc[-1]) 
print("Max accuracy during training with test data: ", max(test_acc[:]))

# plot loss diagram
plt.figure(1)
plt.plot(train_loss, label='Train')
plt.plot(test_loss, label='Test')
plt.xlabel('#epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# plot accuracy diagram
plt.figure(2)
plt.plot(train_acc, label='Train')
plt.plot(test_acc, label='Test')
plt.xlabel('#epochs')
plt.ylabel('accuracy(%)')
plt.legend()
plt.show()

# poor performance, we use dropout as its regulizer
from keras.layers import Dropout
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=25)) # input layer
model.add(Dense(200, activation='relu')) # hidden layer
model.add(Dropout(0.5)) # dropout layer with the probability of 50% to remove neurons randomly 
model.add(Dense(10, activation='softmax')) # output layer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=30, batch_size=64)

loss, acc = model.evaluate(X_test, y_test)
print('\n Test loss: %.2f, \t accuracy: %.2f %%' %(loss, acc))
pred_class = model.predict_classes(X_test)
print("predicted: ", pred_class, "\n true class: ", y_test_org)




    
















