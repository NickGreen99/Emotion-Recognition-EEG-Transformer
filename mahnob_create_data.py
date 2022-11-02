from pyedflib import highlevel
import numpy as np
import os
import re
import pickle
from xml.dom import minidom

# Get number of Subjects (=29)
n_subjects = 0
for root, dirs, files in os.walk("./venv/MAHNOB_HCI/Subjects"):
    for file in files:
        n_subjects += 1

n_movies = 0
# Get number of movies (=20)
for root, dirs, files in os.walk("./venv/MAHNOB_HCI/MediaFiles"):
    for file in files:
        n_movies += 1

# Get all sessions that are complete in the correct order (24 sessions according to paper and dataset manual)
natsort = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', s[0])]  # natural sorting

all_sessions_tuple = []
all_sessions_xml_tuple = []
for root, dirs, files in os.walk("./venv/MAHNOB_HCI/Sessions"):
    for file in files:
        if file.endswith(".bdf"):
            all_sessions_tuple.append((root, file))
        if file.endswith(".xml"):
            all_sessions_xml_tuple.append((root, file))
all_sessions_tuple = sorted(all_sessions_tuple, key=natsort)  # Natural sorting for folders
all_sessions_xml_tuple = sorted(all_sessions_xml_tuple, key=natsort)  # Natural sorting for folders

# Get only the complete and available data
incomplete_data = [3, 9, 12, 15, 16, 26]  # as suggested in the dataset paper
complete_folders = []

n_subjects = n_subjects - 6

sessions = []

count = 0
for sig in all_sessions_tuple:
    if int(sig[1].split('_')[1]) not in incomplete_data:
        sessions.append(sig)
        complete_folders.append(sig[0])
    count += 1

sessions_xml = []
count = 0
for xml in all_sessions_xml_tuple:
    if xml[0] in complete_folders:
        sessions_xml.append(xml)
    count += 1


# Create list of arrays for all signals for all subjects (subjects x movies x channels x time)
# Create list of session data found in the xml files (subjects x movies x emotions)
data = []
data_headers = []
session_data = []

count = 0
for i in range(0, n_subjects):
    temp_data = []
    temp_headers = []
    temp_session_data = []

    for j in range(0, n_movies):
        # EEG data
        file_name = sessions[count][0] + '/' + sessions[count][1]
        signals, signal_headers, header = highlevel.read_edf(file_name)

        # Keep EEG data and drop other physiological signals
        to_drop = [*range(32, len(signals))]
        signals = np.delete(signals, to_drop, 0)
        signal_headers = np.delete(signal_headers, to_drop, 0).tolist()

        temp_data.append(signals)
        temp_headers.append(signal_headers)

        # Session Data (XML)
        file_name = sessions_xml[count][0] + '/' + sessions_xml[count][1]
        file = minidom.parse(file_name)

        emotion_labels = []

        session_tag = file.getElementsByTagName('session')
        emotion_labels.append(session_tag[0].attributes['feltEmo'].value)  # Felt Emotion
        emotion_labels.append(session_tag[0].attributes['feltArsl'].value)  # Felt Arousal
        emotion_labels.append(session_tag[0].attributes['feltVlnc'].value)  # Felt Valence

        temp_session_data.append(emotion_labels)

        count += 1

    data.append(temp_data)
    data_headers.append(temp_headers)
    session_data.append(temp_session_data)

# Store data structure using pickle
with open("data", "wb") as fp:
    pickle.dump(data, fp)

with open("data_headers", "wb") as fp2:
    pickle.dump(data_headers, fp2)

with open("session_data", "wb") as fp3:
    pickle.dump(session_data, fp3)
