from glob import glob
import scipy.signal
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Import DEAP dataset in Python

ch_names = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1',
            'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2',
            'P4', 'P8', 'PO4', 'O2']

fs = 128  # sampling frequency in deap dataset
total_length = 8064  # 63 sec
pretrial = fs * 3  # pretrial baseline
data_length = total_length - pretrial  # useful signal

subjects = 32
trials = 40
electrodes = 32

deap_data = np.zeros((subjects, trials, electrodes, data_length))
deap_labels = np.zeros((subjects, trials, 2))

subjects_file_path = glob('venv/DEAP/data_preprocessed_python/*.dat')
for count, subject in enumerate(subjects_file_path):
    user_load = pickle.load(open(subject, "rb"), encoding='latin')
    deap_data[count] = user_load['data'][0:trials, 0:electrodes, pretrial:pretrial + data_length]
    deap_labels[count] = user_load['labels'][0:trials, 0:2]

# Epoch data (Segmentation)
window = 6
overlap = 6 * 0.5
number_of_segments = int((data_length - (overlap * fs)) / (overlap * fs))
print(number_of_segments)

eeg_data = np.zeros((subjects, trials, number_of_segments, electrodes, window * fs))

count = 0
for epoch in range(0, number_of_segments):
    window_size = int(window * fs)
    eeg_data[0:subjects, 0:trials, epoch] = deap_data[0:subjects, 0:trials, 0:electrodes, count:count + window_size]
    count += int(overlap * fs)


# Get correct number of samples, separate HIGH AROUSAL from LOW AROUSAL and HIGH VALENCE from LOW VALENCE
# LA --> 1-4 --> 0
# HA --> 6-9 --> 1

# LV --> 1-4 --> 0
# HV --> 6-9 --> 1

# Binary
hala_list = []
hvlv_list = []

# 4-class
av_list = []

for i in range(0, subjects):
    # Binary
    hala_trials = []
    hvlv_trials = []

    # 4-class
    av_trials = []
    for j in range(0, trials):
        # Binary
        hala_segments = []
        hvlv_segments = []

        # 4-class
        av_segments = []

        for k in range(0, int(number_of_segments)):
            # Binary
            if (deap_labels[i, j, 1] <= 4) and (deap_labels[i, j, 1] >= 1):
                hala_segments.append(0)  # low arousal
            elif (deap_labels[i, j, 1] >= 6) and (deap_labels[i, j, 1] <= 9):
                hala_segments.append(1)  # high arousal

            if (deap_labels[i, j, 0]) <= 4 and (deap_labels[i, j, 0] >= 1):
                hvlv_segments.append(0)  # low valence
            elif (deap_labels[i, j, 0] >= 6) and (deap_labels[i, j, 0] <= 9):
                hvlv_segments.append(1)  # high valence

            # 4-class
            if (deap_labels[i, j, 1]) <= 4 and (deap_labels[i, j, 1] >= 1) and (deap_labels[i, j, 0] <= 4) and (deap_labels[i, j, 0] >= 1):
                av_segments.append(0)  # low arousal low valence
            elif (deap_labels[i, j, 1]) <= 4 and (deap_labels[i, j, 1] >= 1) and (deap_labels[i, j, 0] <= 9) and (deap_labels[i, j, 0] >= 6):
                av_segments.append(1)  # low arousal high valence
            elif (deap_labels[i, j, 1]) <= 9 and (deap_labels[i, j, 1] >= 6) and (deap_labels[i, j, 0] <= 4) and (deap_labels[i, j, 0] >= 1):
                av_segments.append(2)  # high arousal low valence
            elif (deap_labels[i, j, 1]) <= 9 and (deap_labels[i, j, 1] >= 6) and (deap_labels[i, j, 0] <= 9) and (deap_labels[i, j, 0] >= 6):
                av_segments.append(3)  # high arousal high valence
        hala_trials.append(hala_segments)
        hvlv_trials.append(hvlv_segments)
        av_trials.append(av_segments)
    hala_list.append(hala_trials)
    hvlv_list.append(hvlv_trials)
    av_list.append(av_trials)


# Make sure that labels are within the specified limits

def assign_values_to_labels(label_list):
    x = []
    y = []
    for i in range(0, subjects):
        x_trials = []
        y_trials = []
        for j in range(0, trials):
            # Checks to see if a clip hasn't got a valid movie rating (1 < rating < 4 [low]) or (6 < rating < 9 [high])
            if label_list[i][j]:
                x_epochs = []
                y_epochs = []
                for k in range(0, int(number_of_segments)):
                    x_epochs.append(eeg_data[i, j, k])
                    y_epochs.append(label_list[i][j][k])
                x_trials.append(x_epochs)
                y_trials.append(y_epochs)
        x.append(x_trials)
        y.append(y_trials)

    return x, y


x_hala, y_hala = assign_values_to_labels(hala_list)
x_hvlv, y_hvlv = assign_values_to_labels(hvlv_list)
x_av, y_av = assign_values_to_labels(av_list)


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
    (f, psd) = scipy.signal.welch(eeg, fs, nperseg=fs, window='hamming')
    freq_res = f[1]-f[0]
    # psd shape for one segment and one channel 1*65
    # frequency shape is 1*65, makes sense cause sampling freq is 128 so we can analyse from 0-64hz
    X = np.zeros((clips * epochs, channels, freq_bands))

    # delta band (4-7Hz)
    X[0:clips * epochs, 0:channels, 0] = np.mean(10 * np.log10(psd[0:clips * epochs, 0:channels, 4:8]), axis=2)
    # slow alpha band (8-10Hz)
    X[0:clips * epochs, 0:channels, 1] = np.mean(10 * np.log10(psd[0:clips * epochs, 0:channels, 8:11]), axis=2)
    # alpha band (8-12Hz)
    X[0:clips * epochs, 0:channels, 2] = np.mean(10 * np.log10(psd[0:clips * epochs, 0:channels, 8:13]), axis=2)
    # beta band (13-30Hz)
    X[0:clips * epochs, 0:channels, 3] = np.mean(10 * np.log10(psd[0:clips * epochs, 0:channels, 13:31]), axis=2)
    # gamma band (30-47Hz)
    X[0:clips * epochs, 0:channels, 4] = np.mean(10 * np.log10(psd[0:clips * epochs, 0:channels, 30:48]), axis=2)

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


# Subjects x Brain Region x Data x Electrodes x Frequency band

x_arousal = []
x_valence = []
x_arousal_valence = []
for i in range(0, subjects):
    x_arousal.append(fe(x_hala[i]))
    x_valence.append(fe(x_hvlv[i]))
    x_arousal_valence.append((fe(x_av[i])))

y_arousal = []
y_valence = []
y_arousal_valence = []
for i in range(0, subjects):
    labels_hal = np.array(y_hala[i])
    clips = labels_hal.shape[0]
    epochs = labels_hal.shape[1]
    reshaped_hal = np.reshape(labels_hal, (clips * epochs))
    y_arousal.append(reshaped_hal)

    labels_hvl = np.array(y_hvlv[i])
    clips = labels_hvl.shape[0]
    epochs = labels_hvl.shape[1]
    reshaped_hvl = np.reshape(labels_hvl, (clips * epochs))
    y_valence.append(reshaped_hvl)

    labels_av = np.array(y_av[i])
    clips = labels_av.shape[0]
    epochs = labels_av.shape[1]
    reshaped_av = np.reshape(labels_av, (clips * epochs))
    y_arousal_valence.append(reshaped_av)


def count_data(li):
    ones = 0
    zeros = 0
    twos = 0
    threes = 0
    for i in li:
        zeros = zeros + i.tolist().count(0)
        ones = ones + i.tolist().count(1)
        if max(y_arousal_valence[0]) > 1:
            twos = twos + i.tolist().count(2)
            threes = threes + i.tolist().count(3)
    return zeros, ones, twos, threes


lv, hv, no1, no2 = count_data(y_valence)
la, ha, no3, no4 = count_data(y_arousal)
lalv, lahv, halv, hahv = count_data(y_arousal_valence)

print('Low Valence: ' + str(lv) + ' | ' + 'High valence: ' + str(hv))
print('Low Arousal: ' + str(la) + ' | ' + 'High arousal: ' + str(ha))
print('Low Arousal Low Valence: ' + str(lalv) + ' | ' + 'Low arousal High Valence: ' + str(lahv) + ' | ' +
      'High Arousal Low Valence: ' + str(halv) + ' | ' + 'High arousal High Valence: ' + str(hahv))


# Store data structure using pickle

# Arousal
with open("deap_hala_x", "wb") as fp:
    pickle.dump(x_arousal, fp)

with open("deap_hala_y", "wb") as fp:
    pickle.dump(y_arousal, fp)

# Valence
with open("deap_hvlv_x", "wb") as fp:
    pickle.dump(x_valence, fp)

with open("deap_hvlv_y", "wb") as fp:
    pickle.dump(y_valence, fp)

# Arousal and Valence
with open("deap_av_x", "wb") as fp:
    pickle.dump(x_arousal_valence, fp)

with open("deap_av_y", "wb") as fp:
    pickle.dump(y_arousal_valence, fp)