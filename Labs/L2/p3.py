import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile


t1 = np.linspace(0, 2000, 1600)
x1 = np.sin(2*np.pi*(1/400)*t1)

t2 = np.linspace(0, 3, 300)
x2 = np.sin(2*np.pi*(1/800)*t2)

t3 = np.linspace(0, 0.3, 1600)
x3 = 300 * t3 % 1

t4 = np.linspace(0, 0.3, 5000)
x4 = np.sign(np.sin(t4 * 300))


fs = 44100
sd.play(x1, fs)
sd.wait()

sd.play(x2, fs)
sd.wait()

sd.play(x3, fs)
sd.wait()

sd.play(x4, fs)
sd.wait()


rate = 44100
wavfile.write('semnal_a.wav', rate, x1)


rate_loaded, x_loaded = wavfile.read('semnal_a.wav')
sd.play(x_loaded, rate_loaded)
sd.wait()


fig, axs = plt.subplots(4)
axs[0].plot(t1, x1)
axs[1].plot(t2, x2)
axs[2].plot(t3, x3)
axs[3].plot(t4, x4)

plt.tight_layout()
plt.show()
