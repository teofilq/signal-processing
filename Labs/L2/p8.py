
#8
import numpy as np
import matplotlib.pyplot as plt


Fs = 1000
T = 1
alpha = np.linspace(-np.pi / 2, np.pi / 2, Fs)


x1 = np.sin(alpha)  
x2 = alpha          
x3 = alpha / (1 + (alpha ** 2) / 6)  


error_approx = np.abs(x2 - x1) 
error_pade = np.abs(x3 - x1)   


fig, axs = plt.subplots(3, figsize=(8, 10))
fig.suptitle('Aproximări pentru sin(x)')


axs[0].plot(alpha, x1, label="sin(x) exact")
axs[0].plot(alpha, x2, label="sin(x) ≈ x", linestyle="--")
axs[0].plot(alpha, x3, label="Pade", linestyle=":")
axs[0].set_title('Funcții')
axs[0].legend()
axs[0].grid()


axs[1].semilogy(alpha, error_approx)
axs[1].set_title('Eroare sin(x) ≈ x')
axs[1].grid()


axs[2].semilogy(alpha, error_pade)
axs[2].set_title('Eroare sin(x) Pade')
axs[2].grid()

plt.tight_layout()
plt.show()

