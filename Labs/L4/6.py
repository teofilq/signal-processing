import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import matplotlib.pyplot as plt


rate_a, data_a = wavfile.read('a.wav')
rate_e, data_e = wavfile.read('e.wav')

# sd.play(data_a[:, 0], rate_a)
# sd.wait()  
# sd.play(data_e, rate_e)
# sd.wait()  


data = data_a[:, 1] + data_e[:len(data_a),1]
sd.play(data , rate_a)
n = len(data)
seg_len = int(n*0.01)
overlap = seg_len//2

segments = []

for  i in range(0, n, overlap):
    segment = data[i:i+seg_len]
    if len(segment) == seg_len: 
        segments.append(segment)



segments = np.array(segments)

# Pentru fiecare grup, calculează FFT-ul
fft_segments = []
for elem in segments:
    fft_segments.append(np.abs(np.fft.fft(elem)))

fft_segments = np.array(fft_segments)
print(fft_segments)

# Vector coloana 
fft_matrix = np.array(fft_segments).T  

fft_matrix_log = 10 * np.log10(fft_matrix + 1e-7)  

plt.figure(figsize=(10, 6))
extent = [0, n / rate_a, 0, rate_a / 2]  
plt.imshow(fft_matrix_log, aspect='auto', origin='lower', cmap='viridis', extent=extent)
plt.colorbar(label='dB')
plt.xlabel('Timp')
plt.ylabel('Frecventa')
plt.title('Spectrograma')
plt.show()