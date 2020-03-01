from denis_content_detector import ContentDetector
import numpy as np
import os
from PIL import Image
import math
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import wave
import numpy as np
from scipy.io.wavfile import read
import scipy.fftpack

import argparse

import datetime
# Date
now = datetime.datetime.now()


# See if can use PyScene with a relatively high threshold to just detect the scene changes.
# Then, in each scene, use Entropy to find if this scene is an ad or not.

# Audio Sampling Rate
fs = 48000

# Video Sampling Rate
vfs = 30 # frames/sec

# Local minima
last_local_min = 0

height = 270
width = 480
frameLength = height * width * 3
frame_array = np.zeros((width, height,3), np.float32)

array_r = [[0 for x in range(width)] for y in range(height)] 
array_g = [[0 for x in range(width)] for y in range(height)] 
array_b = [[0 for x in range(width)] for y in range(height)] 
rgbArray = np.zeros((height,width,3), 'uint8')

entropy_arr = []
prev_entropy = 0

sceneDetect_cuts = [] # List of frames at which cuts where found using SceneDetect
entropy_cuts = []
energy_cuts = []

fft_cut_entropies = []

input_video = ''
input_audio = ''

outlog_name = now.strftime("%d-%B-%Y_%H-%M-%S")
# Create experiment dir
dir_name = 'data/' + outlog_name
os.mkdir(dir_name)

# Create entropy-results dir
entropy_dir_name = dir_name + '/entropy'
os.mkdir(entropy_dir_name)

# Create Scene-detection-results dir
sceneDetection_dir_name = dir_name + '/sceneDetection'
os.mkdir(sceneDetection_dir_name)

# Create Energy dir
energy_dir_name = dir_name + '/energy'
os.mkdir(energy_dir_name)

# Create Audio FFT dir
audio_fft_dir_name = dir_name + '/FFT'
os.mkdir(audio_fft_dir_name)

# Create Entropy file
entropy_file_name = entropy_dir_name + '/entropy_' + outlog_name + '.txt'
entropy_file = open(entropy_file_name, "w")

# Create Scene detection cuts file
scene_cuts_file_name = sceneDetection_dir_name + "/scene_cuts_" + outlog_name + ".txt"
scene_cuts_file = open(scene_cuts_file_name, "w")

# Create Energy cuts file
energy_cuts_file_name = energy_dir_name + "/energy_cuts_" + outlog_name + ".txt"
energy_cuts_file = open(energy_cuts_file_name, "w")


def saveFrameAsImage(fileName):
    img = Image.fromarray(rgbArray)
    img.save(fileName)


def processFrame(start, n, entropyEnabled, entropy_threshold):

    global rgbArray, log_file, r_file, g_file, b_file, frame_array, entropy_arr, prev_entropy, entropy_cuts

    print('frame: ', n)

    f = open(input_video, "rb")
    f.seek(start + height*width*3)
    bytes_img = f.read(frameLength)

    Y_array = [0 for x in range(255)]

    for y in range(height):
        for x in range(width):
            ind = y * width + x
            a = 0
            r = bytes_img[ind]
            g = bytes_img[ind+height*width]
            b = bytes_img[ind+height*width*2]

            Y = 0.299*r + 0.587*g + 0.114*b
            Y_int = int(Y)
            if Y_int < 0:
                Y_int = 0
            elif Y_int >= 255:
                Y_int = 254

            # U = 0.492 * (b-Y)
            # V = 0.877 * (r-Y)
            pixel = [r,g,b]
            frame_array[x][y] = pixel
            array_r[y][x] = r
            array_g[y][x] = g
            array_b[y][x] = b
            Y_array[Y_int] += 1

    rgbArray[..., 0] = array_r
    rgbArray[..., 1] = array_g
    rgbArray[..., 2] = array_b

    if entropyEnabled == 1:
        # calculate Entropy
        frame_entropy = 0
        for i in range(len(Y_array)):
            val_prob = Y_array[i] / (width*height)
            if val_prob != 0:
                frame_entropy += val_prob * math.log(val_prob)

        frame_entropy *= -1

        entropy_arr.append(frame_entropy)
        entropy_file.write(str(n))
        entropy_file.write(',')
        entropy_file.write(str(frame_entropy))
        entropy_file.write('\n')

        # Get Entropy difference
        entropyDiff = abs(frame_entropy - prev_entropy)
        if entropyDiff > entropy_threshold:
            print('-> entropy.Cut at: ', str(n))
            entropy_cuts.append(n)
            imgOutName = entropy_dir_name + "/" + str(n) + ".jpeg"
            saveFrameAsImage(imgOutName)

        # Update prev entropy
        prev_entropy = frame_entropy

    f.close()

def fourier_transform(y, num_frames):
    # Do Fourier Transform
    k = np.arange(num_frames)
    T = num_frames / fs
    frq = k / T
    frq = frq[range(num_frames//2)]
    Y = np.fft.fft(y) / num_frames # fft computing and normalization
    Y = Y[range(num_frames//2)]
    return (frq, Y)


# Saves screenshot 
def saveScreenshot(frame_num, fileName):
    global rgbArray, log_file, r_file, g_file, b_file, frame_array, entropy_arr, prev_entropy        

    n = frame_num
    start = n * height*width*3

    f = open("data_test3.rgb", "rb")
    f.seek(start + height*width*3)
    bytes_img = f.read(frameLength)

    Y_array = [0 for x in range(255)]

    for y in range(height):
        for x in range(width):
            ind = y * width + x
            a = 0
            r = bytes_img[ind]
            g = bytes_img[ind+height*width]
            b = bytes_img[ind+height*width*2]
            array_r[y][x] = r
            array_g[y][x] = g
            array_b[y][x] = b

    rgbArray[..., 0] = array_r
    rgbArray[..., 1] = array_g
    rgbArray[..., 2] = array_b

    img = Image.fromarray(rgbArray)
    img.save(fileName)
    f.close()


def energy_transform(audio_frame_size, energy_threshold, y):
    global last_local_min
    e = []
    e_x = []
    for i in range(0, len(y), audio_frame_size):
        sum_energy = 0
        e_x.append(i)
        for j in range(i, i+audio_frame_size-1):
            sum_energy += y[j] ** 2
        mean_e = sum_energy / audio_frame_size
        energy_frame = 20 * math.log( math.sqrt(mean_e), 10)
        e.append(energy_frame)

        if energy_frame < energy_threshold:

            # Further confirmation that this has not already been found.
            valid_energydrop_threshold = 100000
            if i - last_local_min > valid_energydrop_threshold:
                # New energy cut!
                cut_time = i / fs # sec
                cut_time_minutes = int(math.floor(cut_time / 60))
                cut_time_sec = int(math.floor(cut_time % 60))
                print('energy cut frame: ', i, ', time: ', str(cut_time_minutes), ":", cut_time_sec)
                last_local_min = i
                # Save screenshot
                frame_num = int(round(cut_time * vfs))
                # Add to cuts list
                energy_cuts.append(frame_num)
                
                screenshot_name = energy_dir_name + '/energycut_' + str(frame_num) + ".jpeg"
                saveScreenshot(frame_num, screenshot_name)

                energy_cuts_file.write(str(frame_num) + ',' + str(cut_time_minutes)+ ":" + str(cut_time_sec) + "\n")

    return (e_x, e)


# Audio confirmation using Fourier Transform
#   args  @cuts: list of cuts to confirm
# 1) Ad segments might have higher entropies
# 2) Maybe can just look at the peaks. Ads will have more peaks, and the peaks themselves are larger in amplitude.
def confirm_audio(cuts):
    global fft_cut_entropies
    # Import audio
    audio_data = read(input_audio)
    y = np.array(audio_data[1], dtype=float)

    # Iterate through every pair of segments and compare audio Frequency spectrum
    # Sampling rate: 48,000 samples/sec
    
    
    max_freq = 25000
    desired_max_freq = 400
    ratio = max_freq // desired_max_freq

    # record FFT Entropy values of segments
    FFT_entropy_file_name = audio_fft_dir_name + "/segment_entropies.txt"
    fft_entropy_file = open(FFT_entropy_file_name, "w")

    last_audio_frame = 0
    last_frame_fft_entropy = 0
    for i in range(0, len(cuts)):
        video_frame = cuts[i]
        cur_audio_frame = int(round(video_frame / vfs * fs))

        # Compare current segment FFT to previous segment FFT
        num_frames = cur_audio_frame - last_audio_frame
        y_segment = y[last_audio_frame:cur_audio_frame]

        frq, Y = fourier_transform(y_segment, num_frames)

        # Quantize FFT
        # Maybe this is not needed?



        last_video_frame = int(round(last_audio_frame / fs * vfs))

        # Calculate probabilities
        prob_sum = 0
        Y_normalized = abs(Y[range(num_frames//(2*ratio))])
        for amp in Y_normalized:
            value_prob = amp / len(frq)
            prob_sum += value_prob * math.log(value_prob)
        entropy = -1 * prob_sum
        # Normalized entropy
        norm_entropy = entropy / len(Y_normalized)

        fft_cut_entropies.append(entropy)
        print(entropy)

        # Output entropy data
        line = str(last_video_frame) + '-' + str(video_frame) + ',' + str(entropy)
        fft_entropy_file.write(line + '\n')

        # Output FFT data
        FFT_file_name = audio_fft_dir_name + "/fft_" + str(last_video_frame) + "-" + str(video_frame) + ".txt"
        fft_file = open(FFT_file_name, "w")
        for j in range(0, len(Y_normalized)):
            line = str(frq[j]) + ',' + str(Y_normalized[j])
            fft_file.write(line)
            fft_file.write('\n')
        fft_file.close()

        # fig, ax = plt.subplots(1, 1)
        plt.figure()
        plt.plot(frq[range(num_frames//(2*ratio))], abs(Y[range(num_frames//(2*ratio))]),'r') # plotting the spectrum
        plt.xlabel('Freq (Hz)')
        plt.ylabel('|Y(freq)|')
        
        # plt.set_xlabel('Freq (Hz)')
        # plt.set_ylabel('|Y(freq)|')
        
        plot_title = 'FFT from frame ' + str(last_audio_frame/fs*vfs) + ' to ' + str(video_frame) 
        plt.title(plot_title)

        figure_file_name = audio_fft_dir_name + '/FFT_' + str(last_audio_frame/fs*vfs) + '-' + str(video_frame) + '.png'
        plt.savefig(figure_file_name)
        
        
        last_audio_frame = cur_audio_frame
        # if i == 3:
        #     break

    fft_entropy_file.close()



# Analyze Audio
def analyze_audio(startFrame, endFrame):

    # Import audio
    audio_data = read(input_audio)

    y = np.array(audio_data[1], dtype=float)
    actual_audio_length = len(y)                    
    audio_t = int(actual_audio_length / fs)
    dt = 1.0 / fs
    t = np.arange(0, audio_t, dt)
    num_frames = endFrame - startFrame
    print('num frames: ', num_frames)
    y_sample = y[startFrame:endFrame]
    t_sample = t[startFrame:endFrame]

    # Get Energy in dB
    audio_frame_size = 1000
    energy_threshold = 51
    e_x, e = energy_transform(audio_frame_size, energy_threshold, y_sample)

    # Get Frequency transform
    frq, Y = fourier_transform(y_sample, num_frames)

    # Plot Time domain
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(t_sample,y_sample)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')

    # Plot Frequency domain
    # Decide on range to plot
    max_freq = 25000
    desired_max_freq = 400
    ratio = max_freq // desired_max_freq
    ax[1].plot(frq[range(num_frames//(2*ratio))], abs(Y[range(num_frames//(2*ratio))]),'r') # plotting the spectrum
    ax[1].set_xlabel('Freq (Hz)')
    ax[1].set_ylabel('|Y(freq)|')

    # Plot Energy
    ax[2].plot(e_x, e)
    ax[2].set_xlabel('Frame')
    ax[2].set_ylabel('Energy (dB)')

    print(entropy_cuts)
    print(sceneDetect_cuts)
    print(energy_cuts)
    plt.show()



def processVideo(entropyEnabled, sceneDecEnabled, frameStart, frameEnd, scene_threshold, entropy_threshold):
    # Detect advertisements to remove.
    numFrames = int(os.path.getsize(input_video) / (height*width*3))

    contentDetector = ContentDetector(threshold=scene_threshold)
    scene_cuts_file.write('threshold: ' + str(scene_threshold) + '\n')

    for n in range(frameStart, frameEnd):
        start = n * height*width*3
        processFrame(start, n, entropyEnabled, entropy_threshold)

        if sceneDecEnabled == 1:
            cuts = contentDetector.process_frame(n,frame_array)

            # Scene Detection
            if len(cuts) > 0:
                sceneDetect_cuts.append(n)
                print("-> SceneDetect.Cut at: ", n)
                print(cuts)
                imgOutName = sceneDetection_dir_name + "/" + str(n) + ".jpeg"
                saveFrameAsImage(imgOutName)
                scene_cuts_file.write(str(n) + '\n')

    entropy_file.close()

    if entropyEnabled == 1:
        # Plot entropies
        pl.figure()
        xAxis = list(range(frameStart,frameEnd))
        pl.plot(xAxis, entropy_arr)
        pl.title('Entropy by frame')
        pl.xlabel('Frame')
        pl.ylabel('Entropy')
        plt.show()



def main():
    global input_video, input_audio

    parser = argparse.ArgumentParser(description='* Ad Detection *')
    parser.add_argument('-e','--entropy', help='Enable Entropy Detection', required=False, default=1)
    parser.add_argument('-s','--scenedetect', help='Enable Scene Detection', required=False, default=1)
    parser.add_argument('-fs','--framestart', help='Frame start', required=False, default=1)
    parser.add_argument('-fe','--frameend', help='Frame end', required=False, default=4000)
    parser.add_argument('-st','--scenethreshold', help='SceneDetect Threshold', required=False, default=40)
    parser.add_argument('-et','--entropythreshold', help='Entropy Threshold', required=False, default=0.07)
    parser.add_argument('-iv','--inputvideo', help='Input Video File', required=False, default='data_test3.rgb')
    parser.add_argument('-ia','--inputaudio', help='Input audio File', required=False, default='audio.wav')

    args = vars(parser.parse_args())
    print(args)

    enableEntropy = int(args['entropy'])
    enableSceneDetect = int(args['scenedetect'])
    frameStart = int(args['framestart'])
    frameEnd = int(args['frameend'])
    scene_threshold = int(args['scenethreshold'])
    entropy_threshold = float(args['entropythreshold'])
    input_video = str(args['inputvideo'])
    input_audio = str(args['inputaudio'])

    numFrames = int(os.path.getsize(input_video) / (height*width*3))

    if enableEntropy == 1 or enableSceneDetect == 1:
        processVideo(enableEntropy, enableSceneDetect, frameStart, frameEnd, scene_threshold, entropy_threshold)
    fs = 48000
    video_length = 5 * 60 * fs

    # print('entropy cuts:')
    # print(entropy_cuts)

    # 2nd pass: Confirm cuts with audio analysis
    sceneDetect_cuts = [4500, 4693, 4831, 4893, 8493, 8585, 8676, 8806]
    # confirm_audio(sceneDetect_cuts)
    # confirm_audio(entropy_cuts)

    print('sceneDetect cuts:')
    print(sceneDetect_cuts)
    print('respective FFT entropies')
    print(fft_cut_entropies)

    



    # AUDIO ANALYSIS
    analyze_audio(0, 14000000)



# Finds specific frame number of cut using SceneDetect, given approx. time of cut.
# Input
#       @time_cut: approx. time where cut is present.
# Output
#       Frame # of more refined cut.
def refine_cut(time_cut, input_video, scene_threshold=40):
    # Convert time to frame.
    approx_frame = int(round(time_cut * vfs))

    # Search for cut within x frames (30 frames = 1sec)
    x = int(round(vfs + vfs/3))  # 45 frames
    frameStart = int(approx_frame - x)
    frameEnd = int(approx_frame + x)
    found_cuts = []
    print('frame start: ', frameStart)
    print('frame end: ', frameEnd)
    
    # Initialize Detect
    contentDetector = ContentDetector(threshold=scene_threshold)

    f = open(input_video, "rb")
    print('processing frame ')
    # Iterate through frames
    for n in range(frameStart, frameEnd):

        print(n),

        # Read video file
        start = n * height*width*3
        
        f.seek(start + height*width*3)
        bytes_img = f.read(frameLength)

        # First get rgb values into img_array
        img_arr = np.zeros((width, height,3), np.float32)
        for y in range(height):
            for x in range(width):
                ind = y * width + x
                a = 0
                r = bytes_img[ind]
                g = bytes_img[ind+height*width]
                b = bytes_img[ind+height*width*2]
                pixel = [r,g,b]
                img_arr[x][y] = pixel

        # Use SceneDetect to find cuts
        cuts = contentDetector.process_frame(n,img_arr)
        # print(cuts)
        if len(cuts) > 0:
            found_cuts.extend(cuts)

    return found_cuts

if __name__ == '__main__':
    # found_cuts = refine_cut(200, 'data_test1.rgb', 40)
    # print(found_cuts)
    main()

    # saveScreenshot(4370, 'real_ad1.jpeg')

    entropy_file.close()
    scene_cuts_file.close()
    energy_cuts_file.close()