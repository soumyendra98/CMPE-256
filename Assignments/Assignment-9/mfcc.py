# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 13:36:18 2019

@author: cvuppalapati
"""

# MFCC
import os
import matplotlib
#matplotlib.use('Agg') # No pictures displayed 
import pylab
import librosa
import librosa.display
import numpy as np
np.set_printoptions(precision=3)
np.set_printoptions(threshold=12)


# https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html
num_images=5
print("\n PART I \n")

print("\n MFCC \n")
y, sr = librosa.load('002.wav')  
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

print(mfccs)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()
