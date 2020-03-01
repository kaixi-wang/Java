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
from scipy.io import wavfile
from skimage import util
from moviepy.editor import *
# import audiosegment
from scipy.interpolate import interp1d
import math
import main_audio_for_ad_remover

# from sklearn.cluster import KMeans

height = 270
width = 480
energy_cuts, cut_times = main_audio_for_ad_remover.main()


class Audio:
    def __init__(self, filepath, output_dir=None):

        self.source = filepath
        self.name = (filepath.split('/')[-1]).split('.')[0]
        self.audio = AudioFileClip(filepath, fps=48000)
        self.audio.duration = 300.0
        self.duration = self.audio.duration
        self.Fs = 48000  # self.audio.fps
        self.ad_times=None
        self.arr = self.audio.to_soundarray()
        if output_dir is None:
            output_dir =os.path.join('/'.join(self.source.split('/')[:-2]),'output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.saveto=output_dir


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


    def detect_ads(self):                       #doesn't work for dataset3
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
        #padded = util.pad(self.arr, self.padding, mode='constant')
        #blocks = util.view_as_blocks(padded, tuple(self.blocksize, ))
        blocks = util.view_as_blocks(self.arr, tuple(self.blocksize,))
        blocked_dct = np.zeros(blocks.shape[0])
        fft_blocks=np.zeros((blocks.shape[0],24000))
        '''for i in np.arange(blocks.shape[0]):  # - 1):
            blocked_dct[i] = np.sum(np.abs(fftpack.fft(blocks[i][11000:12000])))
            fft_blocks.append(fftpack.fft(blocks[i])
        self.fft=fft_blocks'''
        for i in np.arange(blocks.shape[0]):  # - 1):
            blocked_dct[i] = np.abs(fftpack.dct(blocks[i])[0])
            fft_blocks[i]=np.abs(fftpack.fft(blocks[i]))[0:24000]
        self.dct = blocked_dct
        self.fft=fft_blocks
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
    def remove_ads(self, start_stop_times=None, outputfile=None):
        if start_stop_times is None:
            if self.ad_times is None:
                start_stop_times=self.detect_ads()
            else:
                start_stop_times=self.ad_times
        if type(start_stop_times[0])==int:
            start_stop_times=[start_stop_times]

        num_ads = len(start_stop_times)
        filtered = self.audio.copy()
        removed = []
        for i in np.arange(num_ads):
            removedclip = self.audio.subclip(start_stop_times[i][0], start_stop_times[i][1])
            removed.append(removedclip)
            filtered=filtered.cutout(start_stop_times[i][0],start_stop_times[i][1])
        tt=np.arange(0,filtered.duration, 1.0/48000)
        filtered=filtered.set_start(0)
        self.filtered=filtered
        if outputfile is not None: #must output segments and then compile?
            #filtered.write_audiofile(outputfile, fps=48000, buffersize=1000,nbytes=2, codec='pcm_s16le')
            #audio = np.vstack(audio_clip.iter_frames())
            #filtered_arr=filtered.to_soundarray(tt)
            #new_clip=AudioFileClip(filtered_arr,fps=48000)
            new_clip = np.vstack(self.filtered.iter_frames())
            new_clip.write_audiofile(outputfile, fps=48000, buffersize=2000,nbytes=2, codec='pcm_s16le')

            #wavfile.write(outputfile, 48000, filtered_arr)
        return filtered, removed

    def energy_transform(self, audio_frame_size, energy_threshold,startFrame=None, endFrame=None):
        if startFrame is None:
            startFrame = 0
        if endFrame is None:
            endFrame = len(self.arr)
        last_local_min = 0

        e = []
        e_x = []
        energy_cuts = []
        cut_times = []
        y = self.arr[startFrame:endFrame]
        print(y[0])

        for i in range(0, len(y), audio_frame_size):
            sum_energy = 0
            e_x.append(i)
            for j in range(i, i + audio_frame_size - 1):
                sum_energy += y[j] ** 2
            mean_e = sum_energy / audio_frame_size
            energy_frame = 20 * math.log(math.sqrt(mean_e), 10)
            e.append(energy_frame)

            if energy_frame < energy_threshold:

                valid_energydrop_threshold = 100000
                if i - last_local_min > valid_energydrop_threshold:
                    # New energy cut!
                    cut_time = i / fs  # sec
                    cut_time_minutes = int(math.floor(cut_time / 60))
                    cut_time_sec = int(math.floor(cut_time % 60))
                    print('energy cut frame: ', i, ', time: ', str(cut_time_minutes), ":", cut_time_sec)
                    last_local_min = i
                    # Save screenshot
                    frame_num = int(round(cut_time * vfs))
                    # Add to cuts list
                    cut_times.append(cut_time)
                    energy_cuts.append(frame_num)

        return cut_times, energy_cuts  # energy cuts refers to video frame number

    # Analyze Audio
    def analyze_audio(self, audio_frame_size, energy_threshold, startFrame=None, endFrame=None):
        if startFrame is None:
            startFrame = 0
        if endFrame is None:
            endFrame = len(self.arr)
        # Import audio
        y = np.array(self.arr,dtype=float)
        actual_audio_length = len(y)
        audio_t = int(actual_audio_length / self.Fs)
        dt = 1.0 / self.Fs
        t = np.arange(0, audio_t, dt)
        #num_frames = endFrame - startFrame
        # print('num frames: ', num_frames)
        y_sample = y[startFrame:endFrame]
        # t_sample = t[startFrame:endFrame]

        # Get Energy in dB
        audio_frame_size = 1000
        energy_threshold = 51
        (cut_times,energy_cuts) = self.energy_transform(audio_frame_size, energy_threshold,y_sample)
        #print('Audio cuts (s): ', cut_times)
        #print('Frame cuts (Frame num): ', energy_cuts)
        return cut_times, energy_cuts
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

def read_rgb(filename, height=270, width=480):
    f = open(filename, "r")
    arr = np.fromfile(f, dtype=np.uint8)
    num_frames=int(arr.shape[0]/(height*width*3)) #9000
    arr = arr.reshape(num_frames, (height * width * 3))
    rgb_frames=[]
    for frame_number in np.arange(0, num_frames):

        r = arr[frame_number][0:(width * height)].reshape((height, width))
        g = arr[frame_number][(width * height):(2 * width * height)].reshape((height, width))
        b = arr[frame_number][(2 * width * height):].reshape((height, width))

        # rgb = np.array(list(zip(r, g, b)))
        rgb = np.stack([r, g, b])  # np.dstack((r,g,b))
        rgb_frames.append(rgb.T)
        count += 1
        if count % 500 == 0: print(count)
    return rgb_frames

def A2Vscale(sec):
    return sec*48000


if __name__ == "__main__":

    #INPUT FILES
    wav3='dataset3/Videos/data_test3.wav'
    rgb3='dataset3/Videos/data_test3.rgb'

    #OUTPUT DIRECTORY
    #output_dir='/AdRemoval-master/dataset3/output/'
    output_dir='output/'

    a3 = Audio(wav3)
    a3.detect_ads()
    start_times = [0]

    #energy_cuts, cut_times=main_audio_bad.main()

    filtered=[]
    filtered.append(cut_times[0])
    next_cut_min = filtered[0]
    for i in np.arange(1,len(cut_times)):
            if cut_times[i]-15>next_cut_min:
                next_cut_min=cut_times[i]
                filtered.append(cut_times[i])
    start_times=np.zeros(len(filtered)+1)
    end_times=np.zeros(len(filtered)+1)
    start_times[1:]=filtered

    end_times[:-1]=np.array(filtered)+15
    end_times[-1]=a3.duration
    if int(a3.duration-end_times[-2])>2:
        end_times[-1]=a3.duration
    else:
        end_times[-2]=a3.duration
        end_times = np.array(end_times[:-1])
    timepoints = np.concatenate((start_times, end_times), axis=0)
    timepoints.sort()


    audioclips = []  # remove ads
    for i in np.arange(0,len(timepoints),2):
        if i+1<len(timepoints):
            audioclips.append(a3.audio.subclip(timepoints[i], timepoints[i+1]))

    #save audioclips
    for i,audioclip in enumerate(audioclips):
        audioclip.write_audiofile(os.path.join(output_dir,'audio{}.wav'.format(i)), fps=48000)

    composite_clip = concatenate_audioclips(audioclips)
    composite_clip.write_audiofile(os.path.join(output_dir,'full_audio.wav'), fps=48000)

    #Video
    start_frames = start_times * 30  # frame num = s*fps

    # VIDEO:
    # Read RGB file into array
    f = open(rgb3, "r")
    arr = np.fromfile(f, dtype=np.uint8)
    num_frames=int(arr.shape[0]/(height*width*3)) #9000
    frames = arr.reshape(num_frames, (height * width * 3))

    # Segment video to remove ads
    videoclips=[]
    cut_frames=(np.array(timepoints)*30).astype(int)
    print('Including frames:')
    for i in np.arange(0,len(cut_frames),2):
        if i+1<len(cut_frames):
            print('{} to {}'.format(cut_frames[i],cut_frames[i+1]) )
            clip=frames[cut_frames[i]:cut_frames[i+1],:]
            videoclips.append(clip)
            clip.tofile(os.path.join(output_dir,'video{}.rgb'.format(i)))
    np.vstack(videoclips).tofile(os.path.join(output_dir,'fullvideo.rgb'))

    #videoclips.append(arr[(energy_cuts[0]+450): energy_cuts[3],:])
    #videoclips.append(arr[(energy_cuts[3]+450): ,:])

    '''check cut points: 0=start, 1=data-ad, 2=ad-data, 3=data-ad, 4=ad-data 5=data-ad
    #img2 wrong?
    #img4 wrong?
    
    imgs = []
    prev_img = []
    next_img = []
    for frame in cut_frames:
        imgs.append(getPILframe(rgb3, frame_number=frame))
        if frame > 0:
            prev.append(getPILframe(rgb3, frame_number=frame - 1))
        next_img.append(getPILframe(rgb3, frame_number=frame + 1))
    
    for i in np.arange(len(imgs)):
        imgs[i].show()
        prev_img[i].show()
        next_img[i].show()
    '''

    #plt.figure()
    #plt.plot(e_x,e)
    #o1,o2=a3.analyze_audio(audio_frame_size, energy_threshold)
    #print(cut_times)
    #print(o1)
'''
    #For dataset and dataset2
    a1 = Audio(d1)
    idx1 = a1.detect_ads()
    print(a1.source)
    print('duration: {}'.format(a1.duration))
    print('cut timepoints:', idx1[0])
    a1.audio.subclip(a1.ad_times[0][0], a1.ad_times[0][1]).preview()
    a1.audio.subclip(a1.ad_times[1][0], a1.ad_times[1][1]).preview()
    data1_audio, data1_ad=a1.remove_ads(outputfile=os.path.join(a1.saveto, 'data1_audio_no_ads.wav'))


    a2 = Audio(d2)
    idx2 = a2.detect_ads()
    print(a2.source)
    print('duration: {}'.format(a2.duration))
    print('cut timepoints:', idx2[0])

    print('If timepoints arent 15sec, use the second returned variable to adjust time interval')
    a2.audio.subclip(a2.ad_times[0][0], a2.ad_times[0][1]).preview()
    a2.audio.subclip(a2.ad_times[1][0], a2.ad_times[1][1]).preview()


    a2 = Audio(wav3)
    idx2 = a2.detect_ads()
    print(a2.source)
    print('duration: {}'.format(a2.duration))
    print('cut timepoints:', idx2[0])

    print('If timepoints arent 15sec, use the second returned variable to adjust time interval')
    a2.audio.subclip(a2.ad_times[0][0], a2.ad_times[0][1]).preview()
    a2.audio.subclip(a2.ad_times[1][0], a2.ad_times[1][1]).preview()'''