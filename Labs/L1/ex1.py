import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(3, 2)

t = np.linspace(0, 0.03, int(0.03/0.0005))  
t1 = np.linspace(0, 0.03, int(0.03/0.005))  #  200 Hz->pas de 0.005

x1 = np.cos(520*np.pi*t + np.pi/3)
x2 = np.cos(280*np.pi*t - np.pi/3)
x3 = np.cos(120*np.pi*t + np.pi/3)

x1_sampled = np.cos(520*np.pi*t1 + np.pi/3)
x2_sampled = np.cos(280*np.pi*t1 - np.pi/3)
x3_sampled = np.cos(120*np.pi*t1 + np.pi/3)

axs[0, 0].plot(t, x1)
axs[0, 0].set_title('x(t)')
axs[1, 0].plot(t, x2)
axs[1, 0].set_title('y(t)')
axs[2, 0].plot(t, x3)
axs[2, 0].set_title('z(t)')

axs[0, 1].stem(t1, x1_sampled)
axs[0, 1].set_title('x[n]')
axs[1, 1].stem(t1, x2_sampled)
axs[1, 1].set_title('y[n]')
axs[2, 1].stem(t1, x3_sampled)
axs[2, 1].set_title('z[n]')

plt.tight_layout()
plt.show()
