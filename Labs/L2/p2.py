import matplotlib.pyplot as plt
import numpy as np

Fs = 1000
T = 1
f = 10

t = np.linspace(0, T, Fs)

x1 = np.sin(2 * np.pi * f * t)
x2 = np.sin(2 * np.pi * f * t + np.pi / 4)
x3 = np.sin(2 * np.pi * f * t + np.pi / 3)
x4 = np.sin(2 * np.pi * f * t + np.pi / 2)

z = np.random.normal(0, 1, len(t))
snr_values = [0.1, 1, 10, 100]
signals_with_noise = []

for snr in snr_values:
    norm_x4 = np.linalg.norm(x4)
    norm_z = np.linalg.norm(z)
    gamma = norm_x4 / (np.sqrt(snr) * norm_z)
    noisy_signal = x4 + gamma * z
    signals_with_noise.append(noisy_signal)

plt.figure(figsize=(10, 6))
plt.plot(t, x1, label="faza 0")
plt.plot(t, x2, label="faza pi/4")
plt.plot(t, x3, label="faza pi/3")
plt.plot(t, x4, label="faza pi/2 fără zgomot")
plt.title("Semnale sinusoidale cu faze diferite")
plt.legend()
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(5, 1, figsize=(10, 8))

for i, snr in enumerate(snr_values):
    axs[i].plot(t, signals_with_noise[i], label=f"SNR={snr}")
    axs[i].set_title(f"Semnal cu SNR={snr}")
    axs[i].legend()

axs[4].plot(t, x4, label="faza pi/2 fără zgomot")
plt.tight_layout()
plt.show()
