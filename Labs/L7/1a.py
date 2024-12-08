import numpy as np
import matplotlib.pyplot as plt

def process_function(func):
    n1 = np.linspace(-1, 1, 1000)
    n2 = np.linspace(-1, 1, 1000)
    N1, N2 = np.meshgrid(n1, n2)
    
    X = func(N1, N2)
    Y = np.fft.fft2(X)
    Y = np.fft.fftshift(Y)

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(X, cmap='cividis')
    plt.title("Imaginea")
    
    plt.subplot(1, 2, 2)
    plt.imshow(20 * np.log10(np.abs(Y)), cmap='cividis')
    plt.title("Spectrul ")
    plt.colorbar()
    plt.show()

process_function(lambda n1, n2: np.sin(2 * np.pi * n1 + 3 * np.pi * n2))

process_function(lambda n1, n2: np.sin(4 * np.pi * n1) + np.cos(6 * np.pi * n2))