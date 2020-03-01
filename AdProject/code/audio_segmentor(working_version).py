'''
KW
Detects ads in audio
-Edited arr_in and blocksize compatibility

merge with video
videoclip2 = videoclip.set_audio(my_audioclip)

since time starts at 0, subtract one second
'''

import os
from itertools import groupby
from operator import itemgetter
import scipy.io.wavfile as wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from skimage import util
from moviepy.editor import *
import audiosegment
plt.ion()
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import pywt
import scipy.signal as sg
from sklearn.cluster import KMeans

plt.style.use('ggplot')

def smooth_approx(diff_arr):
    x=np.arange(len(diff_arr))
    f2 = interp1d(x, diff_arr,kind='linear')
    xnew = np.linspace(0, np.max(x), num=len(x), endpoint=True)
    return xnew,f2(xnew)
def group_consecutive(data):
    #data_list=[ x for x in data]
    grps=[]
    for k, g in groupby(enumerate(data), lambda x:x[0]-x[1]):
        grps.append(list(map(itemgetter(1), g)))
    return grps

class Audio:
    def __init__(self, filepath):
        self.source = filepath
        self.name = (filepath.split('/')[-1]).split('.')[0]
        self.audio = AudioFileClip(filepath,fps=48000)
        self.duration = self.audio.duration
        self.Fs = 48000#self.audio.fps
    # returns ad times (in seconds) where ads are detected... needs slight scaling to loss/skew from transformations and differentiating
    def get_peaks(self, sliced_peaks):
        peaks = []
        peak_idx = []
        threshold = np.max(sliced_peaks) * .25
        for i, peak in enumerate(sliced_peaks):
            if np.abs(peak) >= threshold:
                peaks.append(peak)
                peak_idx.append(i)
        return threshold,peaks, peak_idx

    def detect_ads(self):
        print('Detecting ads for {}...'.format(self.name))
        self.arr = self.audio.to_soundarray()[:, 0]
        self.blocksize = np.array([self.Fs][0]).reshape(1, )  # np.array(44100).reshape((-1,1))
        self.w = 8  # window size
        if self.arr.shape[0] % self.Fs != 0:
            #self.arr=audiosegment.from_file(self.source).resample(sample_rate_Hz=48000, sample_width=2, channels=1).to_numpy_array()
            padding = int(np.abs((self.arr.shape[0] % self.Fs - self.Fs)))
            #self.padding = (padding,0)
            self.padding =(padding+self.Fs, self.Fs)
        else:
            #self.padding = 0
            self.padding = (self.Fs,self.Fs)
        padded = util.pad(self.arr, self.padding, mode='constant')
        blocks = util.view_as_blocks(padded, tuple(self.blocksize, ))
        # blocks = util.view_as_blocks(self.arr, tuple(self.blocksize,))
        blocked_dct = np.zeros(blocks.shape[0])
        for i in np.arange(blocks.shape[0]):# - 1):
            blocked_dct[i] = np.abs(fftpack.dct(blocks[i])[0])
        self.dct=blocked_dct
        diff_dct = np.diff(blocked_dct)# ,append=blocked_dct[-2])#,prepend=blocked_dct[1])#,append=blocked_dct[-1])
        plt.figure()
        plt.plot(np.arange(len(diff_dct)), diff_dct)
        diff2_dct = np.diff(diff_dct ,n=2)#,append=blocked_dct[-2])#prepend=diff_dct[1])#,append=diff_dct[-1])
        #plt.figure()
        #plt.plot(np.arange(len(diff2_dct)), diff2_dct)
        x,y=smooth_approx(diff2_dct)
        xthreshold, xpeaks, xpeak_idxs = self.get_peaks(y)
        print('x',xpeak_idxs)
        plt.plot(x,y,'--')
        plt.title('{}: 2nd Order Diff of DCT[0] with block_shape={}'.format(self.name, self.blocksize))
        plt.ylabel('Magnitude of 2nd Order Diff')
        plt.xlabel('Time (s)')
        # padded_diff2_dct=util.pad(diff2_dct,pad_width=self.w,mode='reflect')#'edge')
        # diff2_dct_slices = util.view_as_windows(padded_diff2_dct, window_shape=(self.w,), step=1)
        #diff2_dct_slices = util.view_as_windows(util.pad(diff2_dct,(self.w,0), mode='reflect'), window_shape=(self.w,), step=1)
        diff2_dct_slices = util.view_as_windows(diff2_dct,window_shape=(self.w,), step=1)
        diff2_dct_slices_rev = util.view_as_windows(diff2_dct[::-1], window_shape=(self.w,), step=1)
        #diff2_dct_slices =np.concatenate((diff2_dct_slices_rev,diff2_dct_slices))
        shift=0#blocks.shape[0]-diff2_dct_slices.shape[0]#300-diff2_dct_slices.shape[0]

        sliced_peaks = np.zeros(diff2_dct_slices.shape[0]+shift)
        sliced_peaks_rev = np.zeros(diff2_dct_slices.shape[0]+shift)

        self.Tmax = sliced_peaks.shape[0]
        self.scale = self.Tmax / self.duration
        print('scale (intervals/duration)',self.scale)
        diff2_dct_thresh=np.max(np.abs(diff2_dct))*.2
        for i in np.arange(sliced_peaks.shape[0]-shift):
            sliced_peaks[i+shift] = np.max(diff2_dct_slices[i])
            sliced_peaks_rev[i] = np.max(diff2_dct_slices_rev[i])
            #sliced_peaks[i] = np.max(np.abs(diff2_dct_slices[i]))#-np.min(np.abs(diff2_dct_slices[i])))/np.std((diff2_dct_slices[i]))
            if np.abs(sliced_peaks[i])<diff2_dct_thresh:
                sliced_peaks[i]=0
            if np.abs(sliced_peaks_rev[i])<diff2_dct_thresh:
                sliced_peaks_rev[i]=0
        threshold,peaks, peak_idxs = self.get_peaks(sliced_peaks)
        rthreshold, rpeaks, rpeak_idxs = self.get_peaks(sliced_peaks_rev)
        start_stop_times_rev=[[np.min(x), np.max(x)] for x in group_consecutive(sorted(blocks.shape[0]-np.array(rpeak_idxs)))]
        print('reversed peaks:',start_stop_times_rev)
        combined_peaks=group_consecutive(sorted(np.array(peak_idxs+rpeak_idxs)))
        print('combined peaks:',combined_peaks)

        self.sliced_peaks=sliced_peaks
        self.threshold=threshold
        plt.plot(np.arange(self.w,len(sliced_peaks)+self.w), sliced_peaks)
        print('sliced_peaks shape:', sliced_peaks.shape)
        #plt.scatter(peak_idxs), sliced_peaks[peak_idxs])
        # peaks_win1, idx_win1 = self.get_peaks(padded_diff2_dct)
        # idx=[i / self.scale for i in idx]
        # idx_win1=[i / self.scale for i in idx_win1]
        idxs=group_consecutive(peak_idxs)
        #idxs=[[p/self.scale for p in idx] for idx in idxs]
        start_stop_times=[[np.min(x), np.max(x)] for x in idxs]
        print(idxs)
        self.ad_times=start_stop_times

        return start_stop_times, start_stop_times_rev #use reversed to make adjustments to intervals if needed

    def sec2idx(self,seconds):
        return seconds*self.Fs
    def tsplot(self,y, x=None, hold=None,*args):
        if args:
            label=', '.join(args)
        else:
            label=None
        if x is None:
            x=np.arange(300)
        if hold is None:
            plt.figure()
        plt.title('{}: {}'.format(self.name, label))
        #plt.ylabel('Magnitude of 2nd Order Diff')
        plt.xlabel('Time (s)')
        plt.show()



    '''def verify_intervals(self, start_stop_times):
        #def
        new_intervals=np.zeros(start_stop_times.shape)
        blocks=util.view_as_blocks(self.arr, (self.Fs,))
        X=[]
        for b in blocks:
            X.append(fftpack.fft(b))
        X_magnitude=[np.abs(x) for x in X]
        Xreal=(np.array(X_magnitude)[:,0:24000])
        X1high=X1real[:, 12000:24000]
        X1low=X1real[:, 0:12000]
        Y2=(np.sum(X2high[:,10000:12000],axis=1)*np.std(X2high[:,10000:12000],axis=1))

        for i,interval in enumerate(start_stop_times):
            (s,t)= interval
            if s==0:
                new_intervals[i]=np.array([0,15*self.Fs])
            else:
                if t-s<15:
                    avg=(t-s)/2
                    t2=avg+15
                    s2=avg-15
                x = np.array(blocks[71:95])
                for block in fft[s2:t2]:
                    b, a = sg.butter(4, 500. / (48000 / 2.), 'low')
                    x_fil = sg.filtfilt(b, a, x)
                    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
                    t = np.linspace(0., len(x) / 48000, len(x))
                    ax.plot(t, x, lw=1)'''




if __name__ == "__main__":
    d1 = '/Users/kaixiwang/Documents/USC/CSCI-576/FinalProject/dataset/Videos/data_test1.wav'
    d2 = '/Users/kaixiwang/Documents/USC/CSCI-576/FinalProject/dataset2/Videos/data_test2.wav'
    a1 = Audio(d1)
    idx1 = a1.detect_ads()
    print(a1.source)
    print('duration: {}'.format(a1.duration))
    print('padding: {}'.format(a1.padding))
    print('cut timepoints:',idx1[0])

    a2 = Audio(d2)
    idx2 = a2.detect_ads()
    print(a2.source)
    print('duration: {}'.format(a2.duration))
    print('padding: {}'.format(a2.padding))

    print('cut timepoints:',idx2[0])

    print('If timepoints arent 15sec, use the second returned variable to adjust time interval')