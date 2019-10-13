#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 12:44:31 2019

@author: samaneh
"""

import numpy as np
from scipy import io
from skimage.transform import resize

def load_hoda(train_size=1000, test_size=200, size=5):
    trs = train_size
    tes = test_size
    # load data - matlab file
    dataset = io.loadmat('Data_hoda_full.mat')
    
    # train and test sets
    # squeeze removes single-dimensional entries from the shape of an array
    X_train_org = np.squeeze(dataset['Data'][:trs])
    y_train = np.squeeze(dataset['labels'][:trs])
    X_test_org = np.squeeze(dataset['Data'][trs:trs+tes])
    y_test = np.squeeze(dataset['labels'][trs:trs+tes])
    
    # resize
    X_train_5by5 = [resize(img, (5,5)) for img in X_train_org]
    X_test_5by5 = [resize(img, (5,5)) for img in X_test_org]
    
    # reshape
    X_train = [x.reshape(5*5) for x in X_train_5by5]
    X_test = [x.reshape(5*5) for x in X_test_5by5]
    
    return X_train, y_train, X_test, y_test
    
    
    


