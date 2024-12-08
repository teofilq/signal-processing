import numpy as np
import matplotlib.pyplot as plt


def moving_average_model(signal, errors, q, thetas):
    n = len(signal)
    ma = np.zeros_like(signal)
    for i in range(q, n):
        ma[i] = signal[i] + np.dot(thetas, errors[i-q:i][::-1])
    return ma


t = np.linspace(0, 5, 1000)
trend = t**2
sezon = np.sin(2 * np.pi * 3 * t) + np.sin(2 * np.pi * 7 * t)
zgomot = np.random.normal(0, 0.2, 1000)
serie = trend + sezon + zgomot


q = 3  
epsilon = np.random.normal(0, 0.2, len(serie))
thetas = [1, 0.5, 0.3] 

ma = moving_average_model(serie, epsilon, q, thetas)

plt.plot(t, serie, label='Seria originală')
plt.plot(t, ma, label=f'Model MA(cu orizont {q})', linestyle='--')
plt.legend()
plt.show()
