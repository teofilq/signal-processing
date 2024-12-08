import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0,5,1000)
trend = t**2  
sezon = np.sin(2*np.pi*3*t)+np.sin(2*np.pi*7*t)
zgomot = np.random.normal(0, 0.2, 1000)

serie = trend + sezon + zgomot

p = 10
X = np.array([serie[i-p:i] for i in range(p, len(serie))])
y = serie[p:]
phi = np.linalg.lstsq(X, y, rcond=None)[0]
predicții = np.dot(X, phi)

plt.plot(serie, label='Seria og')
plt.plot(range(p, len(serie)), predicții, label='AR', linestyle='dashed')
plt.legend()
plt.title('AR')
plt.show()
