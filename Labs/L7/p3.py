from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

X = misc.face(gray=True)
pixel_noise = 200
noise = np.random.randint(-pixel_noise, pixel_noise + 1, X.shape)
X_noisy = X + noise

Y = np.fft.fft2(X_noisy)
freq_db = 20 * np.log10(np.abs(Y))
Y[freq_db > 130] = 0
X_denoised = np.real(np.fft.ifft2(Y))

plt.imshow(X, cmap='gray')
plt.title('Original')
plt.show()

plt.imshow(X_noisy, cmap='gray')
plt.title('Noisy')
plt.show()

plt.imshow(X_denoised, cmap='gray')
plt.title('Denoised')
plt.show()
