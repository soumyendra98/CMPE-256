# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 16:41:49 2018

@author: cvuppalapati
"""

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
np.set_printoptions(precision=3)
np.set_printoptions(threshold=sys.maxsize)

# https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html

print("\n PART I \n")

print("\n mel \n")
sig, sample_rate = librosa.load('002.wav')  
## sig, sample_rate = librosa.load('C:\\Hanumayamma\\FEMH\\FEMH Data\\Training Dataset\\Normal\\002.wav')  

mel = librosa.feature.melspectrogram(y=sig, sr=sample_rate)
print(mel)

print("\n mel_mean \n")
mel_mean = np.mean(librosa.feature.melspectrogram(sig, sr=sample_rate).T,axis=0)

print(mel_mean)

print("\n PART II \n")

# using pre-computed spectrogram
print("\n pre-computed spectrogram \n")
D = np.abs(librosa.stft(sig))**2
S = librosa.feature.melspectrogram(S=D)
print(S)

print("\n MEAN pre-computed spectrogram \n")
S_mean = np.mean(librosa.feature.melspectrogram(S=D).T,axis=0)

print(S_mean)

print("\n PART III \n")
 # Passing through arguments to the Mel filters
print("\n Passing through arguments to the Mel filters \n")
new_S = librosa.feature.melspectrogram(y=sig, sr=sample_rate, n_mels=128,fmax=8000)
print(new_S)

print("\n MEAN Passing through arguments to the Mel filters \n")
mean_new_S = np.mean(librosa.feature.melspectrogram(y=sig, sr=sample_rate, n_mels=128,fmax=8000))
print(mean_new_S)


print("\n PART IV  \n")
print("\n chroma \n")

chroma = librosa.feature.chroma_cqt(y=sig, sr=sample_rate)


print(chroma)

print("\n agglomerative cluster \n")

bounds = librosa.segment.agglomerative(chroma, 20)


print(bounds)


bound_times = librosa.frames_to_time(bounds, sr=sample_rate)

print("\n bound_times \n")
print(bound_times)


print("\n PART V  \n")

# ************* Use left-aligned frames, instead of centered frames

stft = np.abs(librosa.stft(sig,center=False))


print('{0:.15f}'.format(sig[len(sig)-1]))

# magnitude spectrogram
magnitude = np.abs(stft)  # (1+n_fft/2, T)

# power spectrogram
power = magnitude ** 2  # (1+n_fft/2, T)

mfccs = np.array(librosa.feature.mfcc(y=sig, sr=sample_rate, n_mfcc=8).T)
mfccs_40 = np.mean(librosa.feature.mfcc(y=sig, sr=sample_rate, n_mfcc=40).T,axis=0)
print("\n mfccs_40 \n")
print(mfccs_40)
chroma = np.array(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T)
chroma_mean = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

print("\n chroma_mean \n")
print(chroma_mean)

#mel = np.array(librosa.feature.melspectrogram(X=sig, sr=sample_rate).T)

print("\n PART VI  \n")
mel_mean = np.mean(librosa.feature.melspectrogram(sig, sr=sample_rate).T,axis=0)

print("\n mel_mean \n")
print(mel_mean)

contrast = np.array(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T)
contrast_mean = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)

print("\n contrast_mean \n")
print(contrast_mean)

tonnetz = np.array(librosa.feature.tonnetz(y=librosa.effects.harmonic(sig), sr=sample_rate).T)
tonnetz_mean = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(sig), sr=sample_rate).T,axis=0)

print("\n tonnetz_mean \n")
print(tonnetz_mean)

#https://librosa.github.io/librosa/generated/librosa.feature.spectral_centroid.html
#Compute the spectral centroid.
cent = librosa.feature.spectral_centroid(y=sig, sr=sample_rate)
print("\n Compute the spectral centroid. \n")
print(cent)


print("\n PART VII  \n")
# From spectrogram input:

S, phase = librosa.magphase(librosa.stft(y=sig,center=False))
cent_mega=librosa.feature.spectral_centroid(S=S)
print("\n Compute the spectral mega centroid. \n")
print(cent_mega)

# Using variable bin center frequencies:
## if_gram, D = librosa.ifgram(sig)
n_fft = 64
 
freqs,if_gram, D = librosa.reassigned_spectrogram(y=sig,sr=sample_rate,n_fft=n_fft)

cent_if_gram = librosa.feature.spectral_centroid(S=np.abs(D), freq=if_gram)
print("\n Compute the spectral mega cent_if_gram centroid. \n")
print(cent_if_gram)


print("\n PART VIII  \n")
# ************* UUse a shorter hop length

stft = np.abs(librosa.stft(sig, hop_length=64))


print('{0:.15f}'.format(sig[len(sig)-1]))

# magnitude spectrogram
magnitude = np.abs(stft)  # (1+n_fft/2, T)

# power spectrogram
power = magnitude ** 2  # (1+n_fft/2, T)

mfccs = np.array(librosa.feature.mfcc(y=sig, sr=sample_rate, n_mfcc=8).T)
mfccs_40 = np.mean(librosa.feature.mfcc(y=sig, sr=sample_rate, n_mfcc=40).T,axis=0)
print("\n mfccs_40 \n")
print(mfccs_40)
chroma = np.array(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T)
chroma_mean = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

print("\n chroma_mean \n")
print(chroma_mean)


print("\n PART IX  \n")
#mel = np.array(librosa.feature.melspectrogram(X=sig, sr=sample_rate).T)

 
mel_mean = np.mean(librosa.feature.melspectrogram(sig, sr=sample_rate).T,axis=0)

print("\n mel_mean \n")
print(mel_mean)

contrast = np.array(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T)
contrast_mean = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)

print("\n PART X  \n")

print("\n contrast_mean \n")
print(contrast_mean)

tonnetz = np.array(librosa.feature.tonnetz(y=librosa.effects.harmonic(sig), sr=sample_rate).T)
tonnetz_mean = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(sig), sr=sample_rate).T,axis=0)

print("\n tonnetz_mean \n")
print(tonnetz_mean)

#https://librosa.github.io/librosa/generated/librosa.feature.spectral_centroid.html
#Compute the spectral centroid.
cent = librosa.feature.spectral_centroid(y=sig, sr=sample_rate)
print("\n Compute the spectral centroid. \n")
print(cent)

print("\n PART XI  \n")
# From spectrogram input:

S, phase = librosa.magphase(librosa.stft(y=sig,hop_length=64))
cent_mega=librosa.feature.spectral_centroid(S=S)
print("\n Compute the spectral mega centroid. \n")
print(cent_mega)

print("\n PART XII  \n")
# Using variable bin center frequencies:
##if_gram, D = librosa.ifgram(sig)

freqs,if_gram, D = librosa.reassigned_spectrogram(y=sig,sr=sample_rate,n_fft=n_fft)

cent_if_gram = librosa.feature.spectral_centroid(S=np.abs(D), freq=if_gram)
print("\n Compute the spectral mega cent_if_gram centroid. \n")
print(cent_if_gram)

print("\n PART XIII  \n")
print("\n Get more components \n")
mfcc = librosa.feature.mfcc(y=sig, sr=sample_rate)
print(mfcc)
mfccs = librosa.feature.mfcc(y=sig, sr=sample_rate, n_mfcc=40)
print(mfccs)