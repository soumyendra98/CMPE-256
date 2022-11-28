# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 11:58:12 2018

@author: cvuppalapati
"""

import os
import matplotlib
matplotlib.use('Agg') # No pictures displayed 
import pylab
import librosa
import librosa.display
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

## sig, fs = librosa.load('C:\\Hanumayamma\\FEMH\\FEMH Data\\Training Dataset\\Normal\\002.wav')   
sig, fs = librosa.load('002.wav')   

# make pictures name 
save_path = 'test.png'

pylab.axis('off') # no axis
pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
S = librosa.feature.melspectrogram(y=sig, sr=fs)
#https://stackoverflow.com/questions/36680402/typeerror-only-length-1-arrays-can-be-converted-to-python-scalars-while-plot-sh/42350817
#vectorize
#print(np.vectorize(S))
librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
pylab.close()