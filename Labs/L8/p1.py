import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0,5,1000)
trend = t**2  
sezon = np.sin(2*np.pi*3*t)+np.sin(2*np.pi*7*t)
zgomot = np.random.normal(0, 0.2, 1000)

serie = trend + sezon + zgomot
plt.plot(t, serie)
plt.show()