import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import matplotlib.pyplot as plt


rate_a, data_a = wavfile.read('a.wav')
rate_e, data_e = wavfile.read('e.wav')

# sd.play(data_a, rate_a)
# sd.wait()  
# sd.play(data_e, rate_e)
# sd.wait()  

data = data_a
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

