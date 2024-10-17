import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(2)
t = np.linspace(0, 8, 10000)
x1 = np.sin(2*np.pi*t)
x2= np.cos(2*np.pi*t+3*np.pi/2)



axs[0].plot(t, x1)  
axs[0].set_title('sin')
axs[1].plot(t, x2)
axs[1].set_title('cos')

plt.tight_layout()
plt.show()