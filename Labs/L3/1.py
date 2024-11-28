import math
import os
import numpy as np
import matplotlib.pyplot as plt

N = 8
F = [[math.e**(-2j * math.pi * m * n / N) for n in range(N)] for m in range(N)]
F1 = np.array(F)
x_imaginary = np.imag(F1)
x_real = np.real(F1)

fig, axs = plt.subplots(N, figsize=(8, 12))
for i in range(N):
    axs[i].plot(x_real[i], label='Partea Reală')
    axs[i].plot(x_imaginary[i], label='Partea Imaginară')
    axs[i].legend()
plt.tight_layout()
output_folder = "output"
plt.savefig(os.path.join(output_folder, "grafic1.png"), format="png", dpi=300)
plt.savefig(os.path.join(output_folder, "grafic1.pdf"), format="pdf")
plt.show()

F_conj_transpose = np.conj(F1).T
FFH = np.matmul(F1, F_conj_transpose)
print(np.allclose(FFH, N * np.eye(N)))
