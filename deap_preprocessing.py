from glob import glob
import scipy.signal
import mne
import pandas
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pickle

subjects_file_path = glob('venv/DEAP/data_preprocessed_python/*.dat')
subjects = len(subjects_file_path)
trials = 40

ch_names = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1',
            'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2',
            'P4', 'P8', 'PO4', 'O2']
sampling_freq = 128  # Hz
ch_types = ['eeg'] * 32
info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)
info.set_montage('standard_1020')

signals = []
labels = []
for subject in subjects_file_path:
    path = Path(subject)
    content = path.read_bytes()
    data = pickle.loads(content, encoding='ISO-8859-1')
    signals.append(data['data'])
    labels.append(data['labels'])


window_size = 6
overlap_size = window_size * 0.5


def read_data(eeg):
    simulated_raw = mne.io.RawArray(eeg, info, verbose='critical')
    epochs = mne.make_fixed_length_epochs(simulated_raw, duration=window_size, overlap=overlap_size, verbose='critical')
    array = epochs.get_data()
    return array


data_length = 60 * sampling_freq
number_of_segments = (data_length - (overlap_size * sampling_freq)) / (overlap_size * sampling_freq)

pre_trial = sampling_freq * 3

# subjects x trials x epochs x channels x samples (32, 40, 20, 32, 768)
eeg_data = np.zeros((subjects, trials, int(number_of_segments), len(ch_names), window_size * sampling_freq))

for i in range(0, subjects):
    for j in range(0, trials):
        eeg_data[i, j] = read_data(signals[i][j][0:32, pre_trial:8064])

# noinspection PyUnresolvedReferences
(f, psd) = scipy.signal.welch(eeg_data, sampling_freq, nperseg=sampling_freq, window='hamming')

freq_bands = 5

X = np.zeros((subjects, trials, int(number_of_segments), len(ch_names), freq_bands))

# delta band (4-7Hz)
X[0:32, 0:40, 0:20, 0:32, 0] = np.mean(psd[0:32, 0:40, 0:20, 0:32, 4:8], axis=4)
# slow alpha band (8-10Hz)
X[0:32, 0:40, 0:20, 0:32, 1] = np.mean(psd[0:32, 0:40, 0:20, 0:32, 8:11], axis=4)
# alpha band (8-12Hz)
X[0:32, 0:40, 0:20, 0:32, 2] = np.mean(psd[0:32, 0:40, 0:20, 0:32, 8:13], axis=4)
# beta band (13-30Hz)
X[0:32, 0:40, 0:20, 0:32, 3] = np.mean(psd[0:32, 0:40, 0:20, 0:32, 13:31], axis=4)
# gamma band (30-47Hz)
X[0:32, 0:40, 0:20, 0:32, 4] = np.mean(psd[0:32, 0:40, 0:20, 0:32, 30:48], axis=4)

# subjects x trials x epochs x labels (valence, arousal)
labels_data_four_class = np.zeros((32, 40, 19))
labels_data_binary_class = np.zeros((32, 40, 19, 2))

for i in range(0, subjects):
    for j in range(0, trials):
        for k in range(0, 19):
            
            # LA, LV --> 0
            # LA, HV --> 1
            # HA, LV --> 2
            # HA, HV --> 3

            if (labels[i][j][1] <= 4) and (labels[i][j][0] <= 4):
                labels_data_four_class[i, j, k] = 0
            elif (labels[i][j][1] <= 4) and (labels[i][j][0] >= 6):
                labels_data_four_class[i, j, k] = 1
            elif (labels[i][j][1] >= 6) and (labels[i][j][0] <= 4):
                labels_data_four_class[i, j, k] = 2
            elif (labels[i][j][1] >= 6) and (labels[i][j][0] >= 6):
                labels_data_four_class[i, j, k] = 3
            else:
                labels_data_four_class[i, j, k] = -1
            
            # LA --> 0
            # HA --> 1
            # LV --> 10
            # HV --> 11
            
            if (labels[i][j][1] <= 4) and (labels[i][j][1] >= 1):
                labels_data_binary_class[i, j, k, 0] = 0  # low arousal
            elif (labels[i][j][1] >= 6) and (labels[i][j][1] <= 9):
                labels_data_binary_class[i, j, k, 0] = 1  # high arousal
            else:
                labels_data_binary_class[i, j, k, 0] = -1

            if (labels[i][j][0]) <= 4 and (labels[i][j][0] >= 1):
                labels_data_binary_class[i, j, k, 1] = 10  # low valence
            elif (labels[i][j][0] >= 6) and (labels[i][j][0] <= 9):
                labels_data_binary_class[i, j, k, 1] = 11  # high valence
            else:
                labels_data_binary_class[i, j, k, 1] = -1
'''
unique, counts = np.unique(labels_data_binary_class, return_counts=True)
l=dict(zip(unique, counts))
print(l)
'''

# Store data structure using pickle
with open("deap_input", "wb") as fp:
    pickle.dump(X, fp)

with open("deap_label_binary", "wb") as fp:
    pickle.dump(labels_data_binary_class, fp)

with open("deap_label_four", "wb") as fp:
    pickle.dump(labels_data_four_class, fp)
