import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile

Fs = 20000
T = 20
f = 10

t = np.linspace(0, T, Fs)
x1 = np.sin(2 * np.pi * f * t)
x2 = np.sin(2 * np.pi * 1.7 * f * t)
x = np.concatenate((x1, x2))

fs = 44100
sd.play(x, fs)
sd.wait()