# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 11:01:23 2016

@author: Riaan
"""

import pandas as pd
import numpy as np

# Load the data from csv
def load():
    paths = ['preprocessed/train_all.csv', 'preprocessed/train.csv', 'preprocessed/valid.csv']
    
    
    TRAIN = pd.read_csv(paths[1])
    VALID = pd.read_csv(paths[2])
    
    return TRAIN, VALID
    
# group the users by ID
def groupusers(TRAIN, VALID, v = False):
    
    strings = TRAIN['CustomerMD5Key'].unique()
    mapping = dict(zip( strings, np.arange(len(strings)) ))
    TRAIN['CustomerMD5Key'] = TRAIN['CustomerMD5Key'].map(mapping)
    TRAIN.sort_values('CustomerMD5Key', ascending=True, inplace = True)
    
    # convenience of saving time
    if v:
        strings = VALID['CustomerMD5Key'].unique()
        mapping = dict(zip( strings, np.arange(len(strings)) ))
        VALID['CustomerMD5Key'] = VALID['CustomerMD5Key'].map(mapping)
        VALID.sort_values('CustomerMD5Key', ascending=True, inplace = True)
        
        return TRAIN, VALID
    
    return TRAIN, None
    
# Normalize continuous 
def normalization(train, valid, normalizeList ):
    train, valid = train, valid
    
    def normalize_train(title):
        mean = train[title].mean()
        std = train[title].std()
        train[title] = (train[title] - mean)/std
        
        return mean, std
    
    def normalize_valid(title, mean, std):
        valid[title]= (valid[title] -mean)/std
        
        
        
    to_normalize = normalizeList
    averages = []
    std = []
    
    # normalize training data
    for title in to_normalize:
        mean_, std_ = normalize_train(title)
        averages.append(mean_)
        std.append(std_)
    
    # normalize validation data
    for key, title in enumerate(to_normalize):
        normalize_valid(title, averages[key], std[key])
        
        
    return train, valid
        
    
def preprocess():
    normalizeList= ['CarAnnualMileage', 'FirstDriverAge', 'CarInsuredValue', 'CarAge', 
                    'VoluntaryExcess', 'PolicyHolderNoClaimDiscountYears',
                    'DaysSinceCarPurchase', 'AllDriversNbConvictions', 'RatedDriverNumber']
    
    print 'Preprocessing'
    print '\tLoading Data'
    train, valid = load()
    
    print '\tGrouping Users'
    train, valid = groupusers(train, valid, v = True)
    
    print '\tNormalizing data'
    train, valid = normalization(train, valid, normalizeList)
    
    return train, valid
    

    
    
    
    
    
    
    
    
preprocess()
    