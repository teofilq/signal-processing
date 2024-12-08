import numpy as np
import matplotlib.pyplot as plt
import scipy

fs = 1000
T = 1
f = 22
N = 100
t = np.linspace(0, T, fs)
s1 = np.sin(2*np.pi*f*t)
srand= np.random.rand(N)

for i in range(3):
    srand = srand*srand
    plt.plot(srand)
    plt.show()
