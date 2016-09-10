# -*- coding: utf-8 -*-
from shutil import copyfile
import datetime
import time

def make_prediction_file(pred):
    # Open a file in write mode
    fo = open("Y_test.predict", "w+")
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
    copyfile('Y_test.predict', 'submission_archive/Y_test.predict' + st)