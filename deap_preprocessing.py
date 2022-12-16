from glob import glob
import scipy.signal
import mne
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pickle
import tensorflow as tf

# EEG data information
ch_names = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1',
            'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2',
            'P4', 'P8', 'PO4', 'O2']
sampling_freq = 128  # Hz
ch_types = ['eeg'] * 32
info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)
info.set_montage('standard_1020')

# EEG segmentation (epoching) function
window_size = 6
overlap_size = window_size * 0.5


def read_data(eeg):
    simulated_raw = mne.io.RawArray(eeg, info, verbose='critical')
    epochs = mne.make_fixed_length_epochs(simulated_raw, duration=window_size, overlap=overlap_size, verbose='ERROR')
    array = epochs.get_data()
    return array


# Read .dat files
subjects_file_path = glob('venv/DEAP/data_preprocessed_python/*.dat')
subjects = len(subjects_file_path)
trials = 40

signals = []
labels = []
for subject in subjects_file_path:
    path = Path(subject)
    content = path.read_bytes()
    data = pickle.loads(content, encoding='ISO-8859-1')
    signals.append(data['data'])
    labels.append(data['labels'])

# Create data matrix (subjects, trials, epochs, channels, samples)
data_length = 60 * sampling_freq
number_of_segments = (data_length - (overlap_size * sampling_freq)) / (overlap_size * sampling_freq)
# We need to erase pre_trial baseline from our data
pre_trial = sampling_freq * 3

# subjects x trials x epochs x channels x samples (32, 40, 20, 32, 768)
eeg_data = np.zeros((subjects, trials, int(number_of_segments), len(ch_names), window_size * sampling_freq))

for i in range(0, subjects):
    for j in range(0, trials):
        eeg_data[i, j] = read_data(signals[i][j][0:32, pre_trial:8064])

# Get correct number of samples, separate HIGH AROUSAL from LOW AROUSAL and HIGH VALENCE from LOW VALENCE

# LA --> 1-4 --> 0
# HA --> 6-9 --> 1

# LV --> 1-4 --> 0
# HV --> 6-9 --> 1

hala_list = []
hvlv_list = []
for i in range(0, subjects):
    hala_trials = []
    hvlv_trials = []
    for j in range(0, trials):
        hala_segments = []
        hvlv_segments = []
        for k in range(0, int(number_of_segments)):
            if (labels[i][j][1] <= 4) and (labels[i][j][1] >= 1):
                hala_segments.append(0)  # low arousal
            elif (labels[i][j][1] >= 6) and (labels[i][j][1] <= 9):
                hala_segments.append(1)  # high arousal

            if (labels[i][j][0]) <= 4 and (labels[i][j][0] >= 1):
                hvlv_segments.append(0)  # low valence
            elif (labels[i][j][0] >= 6) and (labels[i][j][0] <= 9):
                hvlv_segments.append(1)  # high valence
        hala_trials.append(hala_segments)
        hvlv_trials.append(hvlv_segments)
    hala_list.append(hala_trials)
    hvlv_list.append(hvlv_trials)

x_hala = []
y_hala = []
for i in range(0, subjects):
    x_hala_trials = []
    y_hala_trials = []
    for j in range(0, trials):
        if hala_list[i][j]:
            x_hala_epochs = []
            y_hala_epochs = []
            for k in range(0, int(number_of_segments)):
                x_hala_epochs.append(eeg_data[i, j, k])
                y_hala_epochs.append(hala_list[i][j][k])
            x_hala_trials.append(x_hala_epochs)
            y_hala_trials.append(y_hala_epochs)
    x_hala.append(x_hala_trials)
    y_hala.append(y_hala_trials)


# Feature Extraction and data preparations for inputs to HSLT model
def fe(x):
    eeg = np.array(x)
    clips = eeg.shape[0]
    epochs = eeg.shape[1]
    channels = eeg.shape[2]
    samples = eeg.shape[3]
    eeg = np.reshape(eeg, (clips*epochs, channels, samples))
    freq_bands = 5

    # noinspection PyUnresolvedReferences
    (f, psd) = scipy.signal.welch(eeg, sampling_freq, nperseg=sampling_freq, window='hamming')

    X = np.zeros((clips * epochs, channels, freq_bands))

    # delta band (4-7Hz)
    X[0:clips * epochs, 0:channels, 0] = np.mean(psd[0:clips * epochs, 0:channels, 4:8], axis=2)
    # slow alpha band (8-10Hz)
    X[0:clips * epochs, 0:channels, 1] = np.mean(psd[0:clips * epochs, 0:channels, 8:11], axis=2)
    # alpha band (8-12Hz)
    X[0:clips * epochs, 0:channels, 2] = np.mean(psd[0:clips * epochs, 0:channels, 8:13], axis=2)
    # beta band (13-30Hz)
    X[0:clips * epochs, 0:channels, 3] = np.mean(psd[0:clips * epochs, 0:channels, 13:31], axis=2)
    # gamma band (30-47Hz)
    X[0:clips * epochs, 0:channels, 4] = np.mean(psd[0:clips * epochs, 0:channels, 30:48], axis=2)

    x_pf = np.zeros((clips * epochs, 4, 5))
    x_f = np.zeros((clips * epochs, 5, 5))
    x_lt = np.zeros((clips * epochs, 3, 5))
    x_c = np.zeros((clips * epochs, 5, 5))
    x_rt = np.zeros((clips * epochs, 3, 5))
    x_lp = np.zeros((clips * epochs, 3, 5))
    x_p = np.zeros((clips * epochs, 3, 5))
    x_rp = np.zeros((clips * epochs, 3, 5))
    x_o = np.zeros((clips * epochs, 3, 5))
    for i in range(0, clips * epochs):
        x_pf[i] = [X[i, ch_names.index('Fp1')], X[i, ch_names.index('AF3')],
                   X[i, ch_names.index('AF4')], X[i, ch_names.index('Fp2')]]
        x_f[i] = [X[i, ch_names.index('F7')], X[i, ch_names.index('F3')],
                  X[i, ch_names.index('Fz')], X[i, ch_names.index('F4')],
                  X[i, ch_names.index('F8')]]
        x_lt[i] = [X[i, ch_names.index('FC5')], X[i, ch_names.index('T7')],
                   X[i, ch_names.index('CP5')]]
        x_c[i] = [X[i, ch_names.index('FC1')], X[i, ch_names.index('C3')],
                  X[i, ch_names.index('Cz')], X[i, ch_names.index('C4')],
                  X[i, ch_names.index('FC2')]]
        x_rt[i] = [X[i, ch_names.index('FC6')], X[i, ch_names.index('T8')],
                   X[i, ch_names.index('CP6')]]
        x_lp[i] = [X[i, ch_names.index('P7')], X[i, ch_names.index('P3')],
                   X[i, ch_names.index('PO3')]]
        x_p[i] = [X[i, ch_names.index('CP1')], X[i, ch_names.index('Pz')],
                  X[i, ch_names.index('CP2')]]
        x_rp[i] = [X[i, ch_names.index('P8')], X[i, ch_names.index('P4')],
                   X[i, ch_names.index('PO4')]]
        x_o[i] = [X[i, ch_names.index('O1')], X[i, ch_names.index('Oz')],
                  X[i, ch_names.index('O2')]]

    return [x_pf, x_f, x_lt, x_c, x_rt, x_lp, x_p, x_rp, x_o]


x_arousal = []
for i in range(0, subjects):
    x_arousal.append(fe(x_hala[i]))

y_arousal = []
for i in range(0, subjects):
    labels_hal = np.array(y_hala[i])
    clips = labels_hal.shape[0]
    epochs = labels_hal.shape[1]
    reshaped = np.reshape(labels_hal, (clips * epochs))
    y_arousal.append(tf.one_hot(reshaped, 2))

'''
ones = 0
zeros= 0
for i in y_arousal:
    zeros = zeros + i.tolist().count(0)
    ones = ones + i.tolist().count(1)
'''

# Store data structure using pickle
with open("deap_hala_x", "wb") as fp:
    pickle.dump(x_arousal, fp)

# Store data structure using pickle
with open("deap_hala_y", "wb") as fp:
    pickle.dump(y_arousal, fp)
