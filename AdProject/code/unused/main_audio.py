
import wave
import numpy as np
from scipy.io.wavfile import read
import math


# Local minima
last_local_min = 0
# Audio Sampling Rate
fs = 48000
# Video Sampling Rate
vfs = 30 # frames/sec




def energy_transform(audio_frame_size, energy_threshold, y):
    global last_local_min
    e = []
    e_x = []
    energy_cuts = []
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

    return energy_cuts


# Analyze Audio
def analyze_audio(startFrame, endFrame, input_audio):

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
    energy_cuts = energy_transform(audio_frame_size, energy_threshold, y_sample)
    print(energy_cuts)


def main():
    # AUDIO ANALYSIS
    analyze_audio(0, 14000000, 'data_test3.wav')



if __name__ == '__main__':
    main()

