import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0, 5, 1000)
trend = t**2  
sezon = np.sin(2 * np.pi * 3 * t) + np.sin(2 * np.pi * 7 * t)
zgomot = np.random.normal(0, 0.3, 1000)
serie = trend + sezon + zgomot

alpha = 0.3
s = np.zeros_like(serie)
s[0] = serie[0]

for i in range(1, len(serie)):
    s[i] = alpha * serie[i] + (1 - alpha) * s[i - 1]

plt.plot(t, serie, label='Seria originală')
plt.plot(t, s, label='Mediere exponențială', linestyle='--')
plt.legend()
plt.show()
