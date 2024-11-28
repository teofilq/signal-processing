import math 
import matplotlib.pylab as plt
import numpy as np
import os
output_folder = "output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


f = 10
T = 1
Fs = 10000
omega = [3, 7, 15, f]
t = np.linspace(0, T, Fs, endpoint=False)
s = np.sin(2 * np.pi * f * t)

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
n = np.arange(len(t))

for idx, o in enumerate(omega):
    z = [s[i]*math.e ** (-2 * np.pi * 1j * o * i / len(s)) for i in range(len(s))]
    ax = axs[idx // 2][idx % 2]
    sc = ax.scatter(np.real(z), np.imag(z), c=np.abs(z), cmap='viridis', s=1)
    ax.set_aspect('equal')
    ax.set_title(f'ω = {o}')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "grafic2.png"), format="png", dpi=300)
plt.savefig(os.path.join(output_folder, "grafic2.pdf"), format="pdf")
plt.show()
