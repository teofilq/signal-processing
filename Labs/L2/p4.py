import numpy as np
import matplotlib.pyplot as plt

Fs = 1000
T = 1
f = 10

t = np.linspace(0, T, Fs)
x1 = np.sin(2 * np.pi * f * t)
x2 = np.sign(np.sin(t*300))
x_sum = x1 + x2

fig, axs = plt.subplots(3)
fig.suptitle('Semnale')
axs[0].plot(t, x1)
axs[0].set_title('Sinusoida')
axs[1].plot(t, x2)
axs[1].set_title('Semnal dreptunghiular')
axs[2].plot(t, x_sum)
axs[2].set_title('Suma semnalelor')
plt.tight_layout()
plt.show()

