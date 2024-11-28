#ex3
import math 
import matplotlib.pylab as plt
import numpy as np
import os

f = 10
f2 = 7
f3 = 44
fe = 1000
ft = 3
T = 4
N = T * fe

t = np.linspace(0,T,N)
s1 =np.sin(2*np.pi*f*t)
s2 = np.sin(2*np.pi*f2*t+np.pi/5)
s3 = np.cos(2*np.pi*f3*t)
fourier_matrix = np.array([[np.exp(-2j * np.pi * m * n / N) for n in range(N)] for m in range(N)])

s = 2.3*s1+s2+4*s3

sf = np.matmul(fourier_matrix, s)

sfabs = np.abs(sf)

plt.plot(sfabs[:N//12])
output_folder = "output"
plt.savefig(os.path.join(output_folder, "grafic3.png"), format="png", dpi=300)
plt.savefig(os.path.join(output_folder, "grafic3.pdf"), format="pdf")
plt.show()