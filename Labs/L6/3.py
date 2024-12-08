import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows

f = 100
A = 1
phi = 0
fs = 1000
Nw = 200
t = np.linspace(0, Nw / fs, Nw) 
sin_wave = A * np.sin(2 * np.pi * f * t + phi)

rect_windowed = sin_wave * windows.boxcar(Nw)
hann_windowed = sin_wave * windows.hann(Nw)

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(t, sin_wave, label="Sinusoida originală")
plt.title("Semnal original și filtrat")
plt.xlabel("Timp")
plt.ylabel("Amplitudine")
plt.legend()

#Dreptunghiulară
plt.subplot(3, 1, 2)
plt.plot(t, rect_windowed, label="Fereastra dreptunghiulară", color='orange')
plt.xlabel("Timp")
plt.ylabel("Amplitudine")
plt.legend()

#Hanning
plt.subplot(3, 1, 3)
plt.plot(t, hann_windowed, label="Fereastra Hanning", color='green')
plt.xlabel("Timp")
plt.ylabel("Amplitudine")
plt.legend()

plt.tight_layout()
plt.show()