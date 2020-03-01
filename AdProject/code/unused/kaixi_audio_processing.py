'''
KW
Detects ads in audio
-Edited arr_in and blocksize compatibility

merge with video
videoclip2 = videoclip.set_audio(my_audioclip)

play/preview audioclip: (requires pygame)
AudioClip.audio.subclip(start_time,stop_time).preview()


'''

from sys import platform as sys_pf

if sys_pf == 'darwin':
    import matplotlib

    matplotlib.use("TKAgg")
import matplotlib.pyplot as plt

plt.style.use('ggplot')

from itertools import groupby
from operator import itemgetter
import numpy as np
from scipy import fftpack
from skimage import util
from moviepy.editor import *
# import audiosegment
from scipy.interpolate import interp1d


# from sklearn.cluster import KMeans


class Audio:
    def __init__(self, filepath):
        self.source = filepath
        self.name = (filepath.split('/')[-1]).split('.')[0]
        self.audio = AudioFileClip(filepath, fps=48000)
        self.audio.duration = 300.0
        self.duration = self.audio.duration
        self.Fs = 48000  # self.audio.fps

    # returns ad start and end times (in seconds)
    def get_peaks(self, sliced_peaks):
        peaks = []
        peak_idx = []
        threshold = np.max(sliced_peaks) * .25
        consecutive_count=0
        for i, peak in enumerate(sliced_peaks):
            if np.abs(peak) >= threshold:
                consecutive_count+=1
                peaks.append(peak)
                peak_idx.append(i)
                if i - 1 not in peak_idx:
                    pointer = 1
                    while (i - pointer >= 0) and (sliced_peaks[i - pointer] ** 2 > 0):
                        peaks.append(sliced_peaks[i - pointer])
                        peak_idx.append(i - pointer)
                        pointer += 1
            else:
                if consecutive_count>0:
                    if peak>0:
                        peaks.append(peak)
                        peak_idx.append(i)
                consecutive_count=0
        return threshold, peaks, peak_idx

    def detect_ads(self):
        def adjust_time(start_stop_time):
            (t1, t2) = start_stop_time
            if t2 - t1 - 15 > 1:
                t2 = t1 + 15
            elif t2 - t1 <15:
                t2 = t1 + 15
            return [t1, t2]

        print('Detecting ads for {}...'.format(self.name))
        self.arr = self.audio.to_soundarray()[:, 0]
        self.blocksize = np.array([self.Fs][0]).reshape(1, )  # np.array(44100).lreshape((-1,1))
        self.w = 8  # window size
        if self.arr.shape[0] % self.Fs != 0:
            # self.arr=audiosegment.from_file(self.source).resample(sample_rate_Hz=48000, sample_width=2, channels=1).to_numpy_array()
            padding = int(np.abs((self.arr.shape[0] % self.Fs - self.Fs)))
            # self.padding = (padding,0)
            self.padding = (padding + self.Fs, self.Fs)
        else:
            # self.padding = 0
            self.padding = (self.Fs, self.Fs)
        padded = util.pad(self.arr, self.padding, mode='constant')
        blocks = util.view_as_blocks(padded, tuple(self.blocksize, ))
        # blocks = util.view_as_blocks(self.arr, tuple(self.blocksize,))
        blocked_dct = np.zeros(blocks.shape[0])
        for i in np.arange(blocks.shape[0]):  # - 1):
            blocked_dct[i] = np.abs(fftpack.dct(blocks[i])[0])
        self.dct = blocked_dct
        diff_dct = np.diff(blocked_dct)  # ,append=blocked_dct[-2])#,prepend=blocked_dct[1])#,append=blocked_dct[-1])

        diff2_dct = np.diff(diff_dct, n=2)  # ,append=blocked_dct[-2])#prepend=diff_dct[1])#,append=diff_dct[-1])
        x, y = smooth_approx(diff2_dct)
        xthreshold, xpeaks, xpeak_idxs = self.get_peaks(y)

        shift = 0
        diff2_dct_slices = util.view_as_windows(y, window_shape=(self.w,), step=1)
        sliced_peaks = np.zeros(diff2_dct_slices.shape[0] + shift)

        self.Tmax = sliced_peaks.shape[0]
        self.scale = self.Tmax / self.duration
        diff2_dct_thresh = np.max(diff2_dct) * .1
        for i in np.arange(sliced_peaks.shape[0] - shift):
            sliced_peaks[i + shift] = np.max(diff2_dct_slices[i])
            if np.abs(sliced_peaks[i]) < diff2_dct_thresh:
                sliced_peaks[i] = 0
        sliced_peaks = util.pad(sliced_peaks, (self.w, 0), 'symmetric')

        threshold, peaks, peak_idxs = self.get_peaks(sliced_peaks)

        self.sliced_peaks = sliced_peaks
        self.threshold = threshold
        '''plt.figure()
        plt.plot(x, y, ':')
        plt.title('{}: Peaks for smooth_approx(diff2_dct)'.format(self.name))
        plt.ylabel('Magnitude of 2nd Order Diff')
        plt.xlabel('Time (s)')
        plt.plot(np.arange(len(sliced_peaks)), sliced_peaks, '-.')
        plt.scatter(peak_idxs, peaks)'''
        #thresholded = group_consecutive(np.where(self.sliced_peaks > 0)[0].tolist())
        #thresholded = [grp for grp in thresholded if len(grp) >= 15]
        idxs = group_consecutive(peak_idxs)
        start_stop_times = [[np.min(x), np.max(x)] for x in idxs]
        start_stop_times = [adjust_time(x) for x in start_stop_times]
        self.ad_times = start_stop_times
        return start_stop_times#, start_stop_times_rev  # use reversed to make adjustments to intervals if needed

    #
    def remove_ads(self, start_stop_times, outputfile=None):
        num_ads = len(start_stop_times)
        filtered = self.audio.copy()
        removed = []
        for i in np.arange(num_ads):
            filtered.cutout(start_stop_times[i])
            removedclip = self.audio.subclip(start_stop_times[i][0], start_stop_times[i][1])
            removed.append(removedclip)
        if outputfile is not None:
            filtered.write_audiofile(outputfile, fps=48000, nbytes=2, buffersize=2048, codec='pcm_s16le', bitrate=None,
                                     ffmpeg_params=None, write_logfile=False, verbose=True, progress_bar=True)
        return filtered, removed


def smooth_approx(diff_arr):
    x = np.arange(len(diff_arr))
    f2 = interp1d(x, diff_arr, kind='linear')
    xnew = np.linspace(0, np.max(x), num=len(x), endpoint=True)
    return xnew, f2(xnew)


def group_consecutive(data):
    # data_list=[ x for x in data]
    grps = []
    data = list(set(data))
    data.sort()
    for k, g in groupby(enumerate(data), lambda x: x[0] - x[1]):
        grps.append(list(map(itemgetter(1), g)))
    return grps


if __name__ == "__main__":
    d1 = '/Users/kaixiwang/Documents/USC/CSCI-576/FinalProject/dataset/Videos/data_test1.wav'
    d2 = '/Users/kaixiwang/Documents/USC/CSCI-576/FinalProject/dataset2/Videos/data_test2.wav'
    a1 = Audio(d1)
    idx1 = a1.detect_ads()
    print(a1.source)
    print('duration: {}'.format(a1.duration))
    print('cut timepoints:', idx1[0])
    a1.audio.subclip(a1.ad_times[0][0], a1.ad_times[0][1]).preview()

    a2 = Audio(d2)
    idx2 = a2.detect_ads()
    print(a2.source)
    print('duration: {}'.format(a2.duration))
    print('cut timepoints:', idx2[0])

    print('If timepoints arent 15sec, use the second returned variable to adjust time interval')
    a2.audio.subclip(a2.ad_times[0][0], a2.ad_times[0][1]).preview()
    a2.audio.subclip(a2.ad_times[1][0], a2.ad_times[1][1]).preview()
