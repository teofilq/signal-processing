import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(6)

#a)
t1 = np.linspace(0, 2000, 1600)
x1 = np.sin(2*np.pi*(1/400)*t1)

axs[0].plot(t1, x1)


#b)
t2 = np.linspace(0, 3, 300)
x2 = np.sin(2*np.pi*(1/800)*t2)

axs[1].plot(t2, x2)


#c) 
t3 = np.linspace(0, 0.3, 1600) 
x3 = 300*t3%1

axs[2].plot(t3, x3)

#d)
t4 = np.linspace(0, 0.3, 5000)
x4 = np.sign(np.sin(t4*300))
axs[3].plot(t4, x4)

#e)
t5 = np.linspace(0, 1, 50)

plt.tight_layout()
plt.show()