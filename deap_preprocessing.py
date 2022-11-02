import pickle
import scipy.signal
import matplotlib.pyplot as plt

with open('venv/DEAP/data_preprocessed_python/s01.dat', 'rb') as f:
    x = pickle.load(f, encoding='latin1')

plt.plot(x['data'][0][0])
plt.show()

(f, S) = scipy.signal.welch(x['data'][0][0], 128, nperseg=128, window='hamming')

plt.semilogy(f, S)
plt.ylim([1e-7, 1e10])
plt.xlim([0, 128])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()
