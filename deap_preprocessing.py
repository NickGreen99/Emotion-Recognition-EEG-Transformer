import pickle
import scipy.signal
import numpy as np
import os

# Preprocessed data --> downsampled to 128Hz
fs = 128

subjects = 32
channels = 32
clips = 40

# eeg['data'] --> video/trial x channel x data
# eeg['labels'] --> video/trial x label (valence, arousal, dominance, liking)

# Load DEAP dataset
directory = 'venv/DEAP/data_preprocessed_python'
# subjects x clips x electrodes x freq_bands
features = []
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    with open(f, 'rb') as f:
        eeg = pickle.load(f, encoding='latin1')

    # PSD Features using Welch's method

    # Frequency bands
    theta = [4, 7]
    slow_alpha = [8, 10]
    alpha = [8, 12]
    beta = [13, 30]
    gamma = [31, 47]

    freq_bands = [theta, slow_alpha, alpha, beta, gamma]

    # clips x cluster x freq_bands
    total_x = []
    for i in range(0, clips):
        # electrodes x freq_bands
        x = np.zeros(shape=(32, 5))
        # Welch Method for PSD calculations
        for j in range(0, channels):
            (f, S) = scipy.signal.welch(eeg['data'][i][j], fs, nperseg=fs, window='hamming')

            band_count = 0
            for band in freq_bands:
                x[j, band_count] = np.mean(S[min(band):max(band) + 1])
                band_count += 1
        # Division of the Electrode patches to electrode patched clusters
        x_pf = np.stack((x[0], x[1], x[17], x[16]))
        x_f = np.stack((x[3], x[2], x[18], x[19], x[20]))
        x_lt = np.stack((x[4], x[7], x[8]))
        x_c = np.stack((x[5], x[6], x[23], x[24], x[22]))
        x_rt = np.stack((x[21], x[25], x[26]))
        x_lp = np.stack((x[11], x[10], x[12]))
        x_p = np.stack((x[9], x[15], x[27]))
        x_rp = np.stack((x[29], x[28], x[30]))
        x_o = np.stack((x[13], x[14], x[31]))

        total_x.append([x_pf, x_f, x_lt, x_c, x_rt, x_lp, x_p, x_rp, x_o])

    features.append(total_x)

# Store data structure using pickle
with open("deap_input", "wb") as fp:
    pickle.dump(features, fp)