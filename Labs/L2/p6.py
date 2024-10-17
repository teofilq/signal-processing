import numpy as np
import matplotlib.pyplot as plt

Fs = 350
T = 1


t = np.linspace(0, T, Fs)
x1 = np.sin(2 * np.pi * Fs/2 * t)
x2 = np.sin(2 * np.pi * Fs/4 * t)
x3 = np.sin(2 * np.pi * 0 * t)

fig, axs = plt.subplots(3)
fig.suptitle('Semnale')
axs[0].plot(t, x1)

axs[1].plot(t, x2)

axs[2].plot(t, x3)

plt.tight_layout()
plt.show()