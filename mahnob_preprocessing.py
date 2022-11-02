import pickle
import scipy.signal
import matplotlib.pyplot as plt


# Unpickling
with open("data", "rb") as fp:
    data = pickle.load(fp)

with open("data_headers", "rb") as fp2:
    data_headers = pickle.load(fp2)

with open("session_data", "rb") as fp3:
    session = pickle.load(fp3)

###### Maybe we need to 1)downsample 2)BP Filter 3)EOG clean

###### Need to segment EEG data firstly

# PSD features of each EEG segment

'''
(f, S) = scipy.signal.welch(data[0][0][0, :], 256, nperseg=256, window='hamming')

plt.semilogy(f, S)
plt.ylim([1e-7, 1e10])
plt.xlim([0, 128])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()
'''
plt.plot(data[0][1][0,:])
plt.show()