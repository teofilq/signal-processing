import numpy as np
import matplotlib.pyplot as plt

def process_spectrum(spectrum_func):
    N = 100
    spectrum = np.zeros((N, N), dtype=complex)
    spectrum_func(spectrum, N)
    image = np.fft.ifft2(np.fft.ifftshift(spectrum))
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(image), cmap='cividis')
    plt.title("Imaginea")
    
    plt.subplot(1, 2, 2)
    plt.imshow(20 * np.log10(np.abs(spectrum) + 1e-6), cmap='cividis')
    plt.title("Spectrul")
    plt.colorbar()
    plt.show()

process_spectrum(
    lambda Y, N: (Y.__setitem__((0, 5), 1), Y.__setitem__((0, N - 5), 1))
)
process_spectrum(
    lambda Y, N: (Y.__setitem__((5, 0), 1), Y.__setitem__((N - 5, 0), 1))
)
process_spectrum(
    lambda Y, N: (Y.__setitem__((5, 5), 1), Y.__setitem__((N - 5, N - 5), 1))
)