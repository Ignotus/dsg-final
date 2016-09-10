# -*- coding: utf-8 -*-
"""
Voting file 
"""
import os
import numpy as np
from make_prediction_file import make_prediction_file
def vote():
    directory = 'submission_archive/'
    
    result = []
    for filename in os.listdir(directory):
        filename = os.path.join(directory, filename)
        
        with open(filename, 'rb') as f:
            predList = f.read().splitlines() 
            result.append(np.array(predList).astype('float64'))
            #predictions = np.reshape(predictions, (len(predictions), 1))
    result = np.transpose(np.vstack(tuple(result)))
    result =np.average(result, axis = 1)
    print result.shape
    
    make_prediction_file(result)
    
    
            #results.append(np.asarray([float(x) for x in text]))
            
            
            
            
            
        

vote()