import numpy as np
import matplotlib.pyplot as plt


fig, axs = plt.subplots(2)
Fs = 39000
T= 1
f = 4



t = np.linspace(0, T, Fs)
x = np.sin(2*np.pi* f*t)


t1 = np.linspace(0,T, 8)
x1 = np.sin(2*np.pi* f*t1)

axs[0].stem(t1, x1,'red')
axs[0].plot(t,x)

f2 =4+ 7*3 #4 este f initial, 7 este frecventa de esantionare iar k ul este aici 3.
x2 = np.sin(2*np.pi* f2*t)

    
axs[1].stem(t1, x1, 'red')
axs[1].plot(t,x2)
plt.show()