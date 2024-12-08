import time
import numpy as np
import matplotlib.pyplot as plt

val = [128, 256, 512, 1024, 2048, 4096, 8192]
dft_times = []
fft_times = []

for N in val:
    # x = np.random.rand(N)
    t = np.linspace(0, 1, N, endpoint=False)
    x = np.sin(2 * np.pi * 5 * t) 
    start = time.perf_counter()
    F = [[np.exp(-2j * np.pi * m * n / N) for n in range(N)] for m in range(N)]
    F = np.array(F)
    dft_result = np.dot(F, x)
    stop = time.perf_counter()
    dft_times.append(stop - start)
    start = time.perf_counter()
    fft_result = np.fft.fft(x)
    stop = time.perf_counter()
    fft_times.append(stop - start)

    print(f"N = {N}")
    print(f"Timp DFT: {dft_times[-1]}")
    print(f"Timp FFT: {fft_times[-1]}")

plt.plot(val, dft_times, label='dft')
plt.plot(val, fft_times, label='fft')
plt.yscale('log')
plt.xlabel('dimensiunea lui x')
plt.ylabel('timpu')
plt.legend()
plt.show()
