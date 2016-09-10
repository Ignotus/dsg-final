# -*- coding: utf-8 -*-

import datetime
import time
import shutil
import numpy as np

def make_prediction_file(pred):
    # Open a file in write mode
    fo = open("submission/Y_test.predict", "w+")
    print "Name of the file: ", fo.name
    # Write a line at the end of the file.
    for i, p in enumerate(pred):
        if i != len(pred) - 1:
            fo.write(str(p) + "\n")
        else:
            fo.write(str(p))
    
    # Now read complete file from beginning
    # Close opend file
    fo.close()
    
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S')
    shutil.copyfile('submission/Y_test.predict', 'submission_archive/Y_test.predict' + st)
    
    shutil.make_archive('submission', 'zip', 'submission')  

#make_prediction_file(np.zeros(303913))