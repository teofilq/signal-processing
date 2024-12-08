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


print(np.sort(fftsignal_half)[-1:-5:-1])
fft4 = np.argsort(fftsignal_half)[-1:-5:-1]
for i in fft4:
    z = 1/frequencies[i]
    print(i)
    print(z/24)
    print()

#761.91 - creșterea nr. de mașini
#380.95 - fenomene anuale,trafic crescut în periaoda sărbătorilor
#0.99 - cicluri zilnice ale traficului auto, orele de vârf
#253.97 - traficul sezonier

plt.plot(frequencies, fftsignal_half)
plt.xlabel('Frecvența (1/oră)')
plt.ylabel('Amplitudinea')
plt.title('Spectrul semnal')
plt.tight_layout()
plt.grid()
plt.show()


# g) 
dates = data['data'] 
april_mask = np.array([date.endswith('-04-2013') for date in dates]) 
april_signal = cleansignal[april_mask]
time_vector = np.arange(0, len(april_signal)) /24


plt.figure(figsize=(12, 6))
plt.plot(time_vector, april_signal, label="Trafic auto - Aprilie 2013")
plt.xlabel("Timp")
plt.ylabel("Amplitudine semnal")
plt.title("Trafic auto - Aprilie 2013")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# h) Pentru a identifica data de început, analizăm semnalul în funcție de frecvențele detectate (zilnice, săptămânale)
# Comparăm variațiile semnalului (cum ar fi minimul traficului noaptea sau variațiile săptămânale) cu perioadele cunoscute
# și estimăm unde începe primul ciclu complet. Corelăm aceste variații cu zilele calendaristice presupuse

# minusuri:
# Zgomotul și anomaliile pot afecta identificarea
#  Presupune că știm perioadele fenomenelor exact (de ex ciclu zilnic de 24 ore) și le putem compara cu semnalul observat


#i)

#scoatem toate componentele cu frecvența mai mare decat cea a zilelor
# i) Păstrăm doar componentele cu indici mai mici decât 762 care era indexul componentei specifice variației pe zi

max_freq_index = 762
fft_signal = np.fft.fft(cleansignal)
filtered_fft = np.zeros_like(fft_signal, dtype=complex)
filtered_fft[:max_freq_index] = fft_signal[:max_freq_index]
filtered_fft[-max_freq_index:] = fft_signal[-max_freq_index:] 
filtered_signal = np.fft.ifft(filtered_fft).real
time_vector = np.arange(0, len(filtered_signal)) / Fs

plt.figure(figsize=(12, 6))
plt.plot(time_vector, cleansignal, label="Semnal original")
plt.plot(time_vector, filtered_signal, label="Semnal2")
plt.xlabel("Timp")
plt.ylabel("Amplitudine")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()