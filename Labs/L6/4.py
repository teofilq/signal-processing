import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import windows
import scipy.signal as signal

# a) selectez 3 zile
x = np.genfromtxt('Train.csv', delimiter=',', skip_header=1, dtype=None, encoding='utf-8')
data = np.array(x, dtype=[('index', 'i4'), ('data', 'U10'), ('semnal', 'f4')])
semnal = data['semnal'] 
semnal = np.array(semnal)

cleansignal = semnal - semnal.mean()

n = len(cleansignal)

dates = data['data']
days_mask = np.array([date in ['01-04-2013', '02-04-2013', '03-04-2013'] for date in dates])
days_signal = cleansignal[days_mask]
time_vector = np.arange(0, len(days_signal)) / 24

days_signal = days_signal - days_signal.mean()

# b) Media alunecătoare
window_sizes = [5, 9, 13, 17]

plt.figure(figsize=(12, 8))
for w in window_sizes:
    window = np.ones(w) / w
    smoothed = np.convolve(days_signal, window, mode='valid')
    plt.plot(smoothed, label=f"Media alunecatoare (w={w})")

plt.plot(days_signal, label="Semnal original", alpha=0.5, color="gray", linestyle="--")
plt.title("Media alunecătoare ")
plt.xlabel("Timp")
plt.ylabel("Vehicule")
plt.legend()
plt.show()

# c) Frecvența de tăiere
fft = np.fft.fft(days_signal)
fft_abs = [np.abs(elem) for elem in fft]

sampling_frequency = 1  # 1 eșantion pe oră
nyquist_frequency = sampling_frequency / 2
cutoff_frequency = 0.1
normalized_cutoff = cutoff_frequency / nyquist_frequency

print("Frecvența de taiere:", cutoff_frequency)
print("Frecvența normalizata:", normalized_cutoff)

# d) 
butter_b, butter_a = signal.butter(N=5, Wn=normalized_cutoff, btype='low', analog=False)
cheby_b, cheby_a = signal.cheby1(N=5, rp=5, Wn=normalized_cutoff, btype='low', analog=False)

filtered_signal_butter = signal.filtfilt(butter_b, butter_a, days_signal)
filtered_signal_cheby = signal.filtfilt(cheby_b, cheby_a, days_signal)

plt.figure(figsize=(10, 6))
plt.plot(days_signal, label='Semnal brut', alpha=0.7)
plt.plot(filtered_signal_butter, label='Filtru Butterworth', linestyle='--')
plt.plot(filtered_signal_cheby, label='Filtru Chebyshev', linestyle='-.')
plt.legend()
plt.title('Filtrarea Semnalului')
plt.xlabel('Timp (ore)')
plt.ylabel('Amplitudine')
plt.grid()
plt.show()

# e) ce filtru aleg in funție de ce fac
print("Butterworth:  fara distorsiuni.")
print("Chebyshev: rapid.")

# f) teste
order_values = [3, 5, 7]
rp_values = [1, 5, 10]

plt.figure(figsize=(10, 6))

for order in order_values:
    butter_b, butter_a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
    filtered_signal_butter = signal.filtfilt(butter_b, butter_a, days_signal)
    plt.plot(filtered_signal_butter, label=f'Butterworth Ord={order}')

for rp in rp_values:
    cheby_b, cheby_a = signal.cheby1(5, rp, normalized_cutoff, btype='low', analog=False)
    filtered_signal_cheby = signal.filtfilt(cheby_b, cheby_a, days_signal)
    plt.plot(filtered_signal_cheby, label=f'Chebyshev rp={rp} dB', linestyle='-.')

plt.legend()
plt.title('Filtre Butterworth și Chebyshev')
plt.xlabel('Timp')
plt.ylabel('Amplitudine')
plt.grid()
plt.show()