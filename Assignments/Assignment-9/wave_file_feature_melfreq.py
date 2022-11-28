# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 12:18:03 2018

@author: cvuppalapati
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 11:58:12 2018

@author: cvuppalapati
"""

import os
import matplotlib
#matplotlib.use('Agg') # No pictures displayed 
import pylab
import librosa
import librosa.display
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)
# https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html#librosa.feature.melspectrogram
#sig, fs = librosa.load('C:\\Hanumayamma\\FEMH\\FEMH Data\\Training Dataset\\Normal\\002.wav')   
sig, fs = librosa.load('021.wav')   
# make pictures name 
save_path = 'test_melfreq.png'
pylab.axis('off') # no axis
#D =  np.abs(librosa.stft(sig))**2
#D_short = np.abs(librosa.stft(sig, hop_length=64))
D_left = np.abs(librosa.stft(sig, center=False))

#print(D_left)

##np.abs(D_left[f, t]) 

#print(D)
pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
#S = librosa.feature.melspectrogram(y=sig, sr=fs,n_mels=128,fmax=8000)
S = librosa.feature.melspectrogram(S=D_left)
#S = librosa.feature.melspectrogram(y=sig, sr=fs)
#D = np.abs(librosa.stft(sig))**2
#S = librosa.feature.melspectrogram(S=D)
#https://stackoverflow.com/questions/36680402/typeerror-only-length-1-arrays-can-be-converted-to-python-scalars-while-plot-sh/42350817
#vectorize
#print(np.vectorize(S))
#librosa.display.specshow(librosa.power_to_db(S, ref=np.max),y_axis='mel', fmax=8000,
#                         x_axis='time')
#pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
#pylab.close()

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(S,
                                             ref=np.max),
                         y_axis='mel', fmax=8000,
                         x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()

