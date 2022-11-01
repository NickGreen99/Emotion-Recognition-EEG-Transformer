from pyedflib import highlevel
import numpy as np
import os
import re
import pickle

# Get number of Subjects (=29)
n_subjects = 0
for root, dirs, files in os.walk("./venv/Subjects"):
    for file in files:
        n_subjects += 1

n_movies = 0
# Get number of movies (=20)
for root, dirs, files in os.walk("./venv/MediaFiles"):
    for file in files:
        n_movies += 1

# Get all sessions that are complete in the correct order (24 sessions according to paper and dataset manual)
natsort = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', s[1])]  # natural sorting

all_sessions_tuple = []
for root, dirs, files in os.walk("./venv/Sessions"):
    for file in files:
        if file.endswith(".bdf"):
            all_sessions_tuple.append((root,file))
all_sessions_tuple = sorted(all_sessions_tuple, key=natsort)

incomplete_data = [3, 9, 12, 15, 16, 26]

n_subjects = n_subjects - 6

sessions = []

count = 0
for sig in all_sessions_tuple:
    if int(sig[1].split('_')[1]) not in incomplete_data:
        sessions.append(sig)
    count += 1

# Create list of arrays for all signals for all subjects (subjects x movies x channels x time)
data = []
data_headers = []

count = 0
for i in range(0, n_subjects):
    temp_data = []
    temp_headers = []
    for j in range(0, n_movies):
        file_name = sessions[count][0] + '/' + sessions[count][1]
        signals, signal_headers, header = highlevel.read_edf(file_name)

        # Keep EEG data and drop other physiological signals
        to_drop = [*range(32, len(signals))]
        signals = np.delete(signals, to_drop, 0)
        signal_headers = np.delete(signal_headers, to_drop, 0).tolist()

        temp_data.append(signals)
        temp_headers.append(signal_headers)
        count += 1

    data.append(temp_data)
    data_headers.append(temp_headers)

# Store data structure using pickle
with open("data", "wb") as fp:
    pickle.dump(data, fp)

with open("data_headers", "wb") as fp2:
    pickle.dump(data_headers, fp2)
