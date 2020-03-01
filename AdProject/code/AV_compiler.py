import moviepy.editor as mpy
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
import time
import cv2
from sys import platform as sys_pf

if sys_pf == 'darwin':
    import matplotlib

    matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
plt.style.use('ggplot')

height = 270
width = 480

full_vid_rgb='/output/fullvideo.rgb'
full_aud_wav='/output/full_audio.wav'
filename = full_vid_rgb
f = open(filename, "r")
arr = np.fromfile(f, dtype=np.uint8)
num_frames = int(arr.shape[0] / (height * width * 3))  # 9000
arr = arr.reshape(num_frames, (height * width * 3))
cv2imgs = []
count = 0
label = filename.split('/')[-1].split('.')[0]
rgb_frames = []
for frame_number in np.arange(0, num_frames):
    cv2img = None
    r = arr[frame_number][0:(width * height)].reshape((height, width))
    g = arr[frame_number][(width * height):(2 * width * height)].reshape((height, width))
    b = arr[frame_number][(2 * width * height):].reshape((height, width))

    # rgb = np.array(list(zip(r, g, b)))
    rgb = np.stack([r, g, b])  # np.dstack((r,g,b))
    rgb_frames.append(rgb.T)
    count += 1
    if count % 500 == 0: print(count)


clip = ImageSequenceClip(rgb_frames, fps=30)
clip.write_videofile(out_vid, fps=30, codec='rawvideo',audio=full_aud_wav, audio_fps=48000, preset='faster', audio_nbytes=2, audio_codec='pcm_s16le', audio_bufsize=2000, temp_audiofile=None, rewrite_audio=True, remove_temp=True, write_logfile=False, verbose=True)

#Preview video
clip.rotate(270).preview()