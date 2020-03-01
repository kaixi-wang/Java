'''
KW
Detects ads in audio
-Edited arr_in and blocksize compatibility
'''

import os
from operator import itemgetter
import scipy.io.wavfile as wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from skimage import util
from moviepy.editor import *
from scipy.signal import find_peaks

plt.style.use('ggplot')

class Audio:
    def __init__(self):
        self.source=None
        #self.name=(filepath.split('/')[-1]).split('.')[0]
        #self.audio=AudioFileClip(filepath)
        self.fs=44100
    #returns ad times (in seconds) where ads are detected... needs slight scaling to loss/skew from transformations and differentiating
    def get_peaks(self, sliced_peaks):
        peaks = []
        peak_idx = []
        threshold = np.max(sliced_peaks) * .4
        for i, peak in enumerate(sliced_peaks):
            if peak >= threshold:
                peaks.append(peak)
                peak_idx.append(i)
        return peaks,peak_idx
    def detect_ads(self, filepath):
        #filepath=avfiles[0]
        self.source=(filepath.split('/')[-1]).split('.')[0]
        self.audio = AudioFileClip(filepath)
        self.arr = self.audio.to_soundarray()[:,0]
        self.blocksize=np.array([44100][0]).reshape(1,)#np.array(44100).reshape((-1,1))

        blocks = util.view_as_blocks(self.arr, tuple(self.blocksize,))
        blocked_dct = np.zeros(blocks.shape[0])
        for i in np.arange(blocks.shape[0] - 1):
            blocked_dct[i] = np.abs(fftpack.dct(blocks[i])[0])
        diff_dct = np.diff(blocked_dct)
        diff2_dct = np.diff(diff_dct)
        self.w = 14 #window size
        diff2_dct_slices = util.view_as_windows(diff2_dct, window_shape=(self.w,), step=1)
        sliced_peaks = np.zeros(diff2_dct_slices.shape[0])
        for i in np.arange(sliced_peaks.shape[0]):
            sliced_peaks[i] = np.max(diff2_dct_slices[i])
        #plt.plot(np.arange(len(sliced_peaks)), sliced_peaks)
        peaks,idx=self.get_peaks(sliced_peaks)
        self.ads=idx
        return idx,peaks

    def plot_dct(self, data, overlay=None, *args):
        #for arg in args:
        #    label
        if overlay is None:
            plt.figure()
        plt.plot(np.arange(len(blocked_dct) - 1), diff_dct)
        plt.title('{} Diff DCT[0] with block_shape=({},)'.format(self.source,self.blocksize))
        plt.ylabel('First DCT Coefficient')
        plt.xlabel('Time (s)')

    def rescale(self):