import wave
import numpy as np
from scipy.io.wavfile import read

import matplotlib
import matplotlib.pyplot as plt

a = read("audio.wav")
print (a)
b = np.array(a[1], dtype=float)

# Data for plotting
t = np.arange(0, len(b), 1)

fig, ax = plt.subplots()
ax.plot(t, b)

ax.set(xlabel='frame', ylabel='volumn',
       title='Frame and volumn')
ax.grid()

# fig.savefig("test.png")
plt.show()

# array([ 128.,  128.,  128., ...,  128.,  128.,  128.])

# wav = wave.open("dataset/Ads/Subway_Ad_15s.wav", 'r')

# print(wav.getnchannels())

# print(wav.getframerate())


# Wav file
# 100 0 98 0 96 0 94 0 92 0
# 5 Hz Audio with amplitude 100