# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 15:05:44 2016

@author: Riaan
"""

import numpy as np
import pandas as pd
from riaan_preprocessing import preprocess, normalization
import os
import cPickle as pc


def frequency(train):
    train['Counts'] = train['CustomerMD5Key'].map(train['CustomerMD5Key'].value_counts())

    return train


def grouped_data(array):
    
    unique = sorted(array['CustomerMD5Key'].value_counts().index.tolist())
    data_matrix = []
    target_matrix = []
    
    # collect data with all the unique id's
    for x in unique:
        
        mat = array.loc[array['CustomerMD5Key'] ==x]
        if mat.empty:
            break
        
        
        # create the target values
        tgt = mat['Label'].values
        target_matrix.append(tgt)
        
        # drop the labels row and append it to the data matrix
        mat = mat.drop('Label', axis=1, inplace=True)
        
        data_matrix.append(mat.values)
        
        if x % 100 == 0:
            print '\t\tWe are @: ', x, '!'
            
    
    return data_matrix, target_matrix 
    
def train_context_DNN(freq = False, save = False, group = True, group_load = True):
    
    # do the preprocessing
    if save and not group_load:
        train, valid = preprocess()
        if not os.path.exists('preprocessedRiaan'):
            os.mkdir('preprocessedRiaan')
        
        print '\tSaving Data'
        for path, df in zip(['train.csv', 'valid.csv'],[train, valid]):
            df.to_csv(os.path.join('preprocessedRiaan', path))
    else:
        print 'Loading Data'
        train = pd.read_csv('preprocessedRiaan/train.csv')
        valid = pd.read_csv('preprocessedRiaan/valid.csv')
    
    # Frequency calculation
    if freq:
        
        print 'Calculating Frequencies'
        train= frequency(train)
        valid = frequency(valid)
        
        train, valid = normalization(train, valid, ['Counts'])
    
    # Grouping
    if group:
        print '\t Grouping Data'
        
        train_data, train_target = grouped_data(train[0:1000])
        valid_data, valid_target = grouped_data(valid[0:1000])
        
        print '\t\t validation Data'
        pc.dump( train_data, open( "preprocessedRiaan/train_data.p", "wb" ) )
        pc.dump(train_target , open("preprocessedRiaan/train_target.p", "wb"))
        
        pc.dump(valid_data, open( "preprocessedRiaan/valid_data.p", "wb" ))
        pc.dump(valid_target , open("preprocessedRiaan/valid_target.p", "wb"))
        
    if group_load and not group:
        
        print '\t\t Loading grouped data'
        train_data = pc.load( open( "preprocessedRiaan/train_data.p", "rb" ) )
        train_target = pc.load( open( "preprocessedRiaan/train_target.p", "rb" ))
        
        valid_data = pc.load( open( "preprocessedRiaan/valid_data.p", "rb" ))
        valid_target = pc.load( open( "preprocessedRiaan/valid_target.p", "rb" ))
        
        for value in train_data:
            print value
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
    
    
    
    
    
train_context_DNN()