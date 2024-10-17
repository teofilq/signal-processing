import matplotlib.pyplot as plt
import numpy as np


Fs = 1000
T = 1
f = 42

t = np.linspace(0, T, Fs)
x1 = np.sin(2 * np.pi * f * t)
x1_decimated = x1[::4]
t1_decimated = t[::4]

x2_decimated = x1[1::4]
t2_decimated = t[1::4]

fig, axs = plt.subplots(3)
fig.suptitle('Semnale')
axs[0].plot(t, x1)
axs[0].set_title('Sinusoida')
axs[1].plot(t1_decimated, x1_decimated)
axs[1].set_title('Decimat cu factorul 4')
axs[2].plot(t2_decimated, x2_decimated)
axs[2].set_title('Decimat cu factorul 4 si shiftat cu 1')


plt.tight_layout()
plt.show()