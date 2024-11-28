import matplotlib.pyplot as plt
import numpy as np



x = np.genfromtxt('Train.csv', delimiter=',', skip_header=1, dtype=None, encoding='utf-8')
data = np.array(x, dtype=[('index', 'i4'), ('data', 'U10'), ('semnal', 'f4')])
signal= data['semnal']
signal = np.array(signal)

cleansignal = signal - signal.mean()
n = len(cleansignal)

#a) 
print("a) frecvența de eșantionare este 1/ora" )
#b)
print(f"b) intervalul este {str(data['data'][1])} -- { str(data['data'][-1])}")
#c)
print("c) daca semanlul a fost esantionat corect fara aliere vom putea verifica frecvențe de pana la 0.5/h(fe/2)")
#d)
fftsignal = np.abs(np.fft.fft(cleansignal))
Fs = 1  # 1 esantion pe oră
frequencies = np.linspace(0, Fs / 2, len(fftsignal) // 2)
fftsignal_half = fftsignal[:len(fftsignal) // 2]

plt.plot(frequencies, fftsignal_half)
plt.xlabel('Frecvența (1/oră)')
plt.ylabel('Amplitudinea')
plt.title('Spectrul semnal')
plt.show()