import numpy as np
import matplotlib.pyplot as plt
import scipy

np.random.seed(42)

N = 5
coef1 = np.random.rand(N + 1)
coef2 = np.random.rand(N + 1)

poly1 = np.poly1d(coef1)
poly2 = np.poly1d(coef2)

print("Polinomul 1:", poly1)
print("Polinomul 2:", poly2)
product_convolution = np.convolve(coef1, coef2)
print(product_convolution)

len_result = len(coef1) + len(coef2) - 1
fft1 = np.fft.fft(coef1, len_result)
fft2 = np.fft.fft(coef2, len_result)

fft_product = fft1 * fft2

product_fft = np.fft.ifft(fft_product).real
print("\nprodusul prin FFT:", product_fft)

assert np.allclose(product_convolution, product_fft), "eww"

print("\nPolinomul(conv):")
print(np.poly1d(product_convolution))



