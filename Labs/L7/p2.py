from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt

X = misc.face(gray=True)
Y = np.fft.fft2(X)
freq_db = 20 * np.log10(np.abs(Y) + 1e-6)

snr_threshold = 50
max_freq = np.max(freq_db)
cutoff_threshold = max_freq - snr_threshold

Y_compressed = Y.copy()
Y_compressed[freq_db < cutoff_threshold] = 0

X_compressed = np.real(np.fft.ifft2(Y_compressed))

plt.imshow(np.abs(X_compressed), cmap=plt.cm.gray)
plt.show()

plt.imshow(20 * np.log10(np.abs(np.fft.fftshift(Y)) + 1e-6), cmap=plt.cm.viridis)
plt.colorbar()
plt.show()
plt.imshow(20 * np.log10(np.abs(np.fft.fftshift(Y_compressed)) + 1e-6), cmap=plt.cm.viridis)
plt.colorbar()
plt.show()
