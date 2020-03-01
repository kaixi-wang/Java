from pyAudioAnalysis import audioTrainTest as aT

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt

from pyAudioAnalysis import audioSegmentation as aS
[flagsInd, classesAll, acc, CM] = aS.mtFileClassification(audiofile, "data/svmSM", "svm", True, 'data/scottish.segments')

from scipy import fftpack

audiofile='/Users/kaixiwang/Documents/USC/CSCI-576/FinalProject/dataset2/Videos/data_test2.wav'
adaudio=['/Users/kaixiwang/Documents/USC/CSCI-576/FinalProject/dataset/Ads/Subway_Ad_15s.wav','/Users/kaixiwang/Documents/USC/CSCI-576/FinalProject/dataset/Ads/Starbucks_Ad_15s.wav']
[Fs, x] = audioBasicIO.readAudioFile(audiofile);
X = fftpack.fft(x)
freqs = fftpack.fftfreq(len(x)) * Fs
fig, ax = plt.subplots()
ax.stem(freqs, np.abs(X))
ax.set_xlabel('Frequency in Hertz [Hz]')
ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
ax.set_xlim(-Fs / 2, Fs / 2)
ax.set_ylim(-5, 110)

[Fs, x] = audioBasicIO.readAudioFile(audiofile);
F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs);
plt.subplot(2,1,1); plt.plot(F[0,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[0]);
plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[1]); plt.show()

#========================
n = len(channel1)
fourier=fft.fft(ad1)


fourier = fourier[0:int(n/2)]

# scale by the number of points so that the magnitude does not depend on the length
fourier = fourier / float(n)

#calculate the frequency at each point in Hz
freqArray = np.arange(0, (n/2), 1.0) * (rate*1.0/n);
plt.figure(1, figsize=(8,6))

plt.plot(np.arange(len(10*np.log10(fourier))), 10*np.log10(fourier), color='#ff7f00', linewidth=0.02)
plt.xlabel('Frequency (kHz)')
plt.ylabel('Power (dB)')
#====================================

aT.featureAndTrain(["classifierData/music","classifierData/speech"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmSMtemp", False)
aT.fileClassification(audiofile, "svmSMtemp","svm")
Result:
(0.0, array([ 0.90156761,  0.09843239]), ['music', 'speech'])

# sampling a sine wave programmatically
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

rate, audio = wavfile.read(filepath)
audio = np.mean(audio, axis=1)

# sampling information
Fs = 44100 # sample rate
T = 1/Fs # sampling period
t = 0.1 # seconds of sampling
N = Fs*t # total points in signal

# signal information
freq = 100 # in hertz, the desired natural frequency
omega = 2*np.pi*freq # angular frequency for sine waves

t_vec = np.arange(N)*T # time vector for plotting
y = np.sin(omega*t_vec)

plt.plot(t_vec,y)
plt.show()

# fourier transform and frequency domain
#
Y_k = np.fft.fft(y)[0:int(N/2)]/N # FFT function from numpy
Y_k[1:] = 2*Y_k[1:] # need to take the single-sided spectrum only
Pxx = np.abs(Y_k) # be sure to get rid of imaginary part

f = Fs*np.arange((N/2))/N; # frequency vector

# plotting
fig,ax = plt.subplots()
plt.plot(f,Pxx,linewidth=5)
ax.set_xscale('log')
ax.set_yscale('log')
plt.ylabel('Amplitude')
plt.xlabel('Frequency [Hz]')
plt.show()




# Kmeans
np.fft.fft(Xad)
fft_mag_ad=np.absolute(fftAD)
kmeans_ad = KMeans(n_clusters=8, random_state=0).fit(fft_mag_ad.reshape(-1,1))
kmeans_ad.cluster_centers_
'''Out[70]: 
array([[ 3182524.78264615],
       [38069387.18242852],
       [  416436.7183776 ],
       [13076264.61689949],
       [95324167.05995426],
       [ 6915254.13654029],
       [23494793.81500596],
       [59212857.76689355]])'''


model = sklearn.cluster.KMeans(n_clusters=2)
labels = model.fit_predict(features_scaled)
plt.scatter(features_scaled[labels==0,0], features_scaled[labels==0,1], c='b')
plt.scatter(features_scaled[labels==1,0], features_scaled[labels==1,1], c='r')
plt.xlabel('Zero Crossing Rate (scaled)')
plt.ylabel('Energy (scaled)')
plt.legend(('Class 0', 'Class 1'))


# TODO: quantize to seconds: split into intervals of size len(samples)/(len(samples)/Fs)
#np.array_split(fftAD, len(fftAD)/(len(fftAD)/Fs) #48000 intervals
fft_mag_ad=np.absolute(fftAD)
qfftAD=np.array_split(fft_mag_ad, int(np.round(len(fftAD)/Fs,0)))

fft_data1_mag=np.absolute(fft_data1)
qfft_data1=np.array_split(fft_data1_mag, int(np.round(len(fft_data1)/Fs)))

quant_kmeans_ad = np.zeros((len(qfftAD), 2))
quant_kmeans_data1 = np.zeros((len(qfft_data1), 2))
for i, q in enumerate(qfftAD):
    model = KMeans(n_clusters=2, random_state=0).fit(fft_mag_ad.reshape(-1, 1))
    centers = model.cluster_centers_
    quant_kmeans_ad[i] = centers

model = KMeans(n_clusters=2, random_state=0)
for i, q in enumerate(qfft_data1):
    model.fit(q.reshape(-1, 1))
    centers = model.cluster_centers_
    quant_kmeans_data1[i] = centers.reshape(1, 2)

'''
len(fftAD)/Fs
Out[78]: 14.998
len(fft_data)/Fs
Out[79]: 299.998'''

np.savetxt("Users/kaixiwang/Documents/USC/CSCI-576/FinalProject/processed/quantified_kmeans_audio_data1.txt.csv", quant_kmeans_data1, delimiter=",")
