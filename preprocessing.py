import pyedflib
import numpy as np

file_name = r'C:\Users\npras\Desktop\Thesis\Work-in-Progress\Datasets\Sessions\2\Part_1_S_Trial1_emotion.bdf'
f = pyedflib.EdfReader(file_name)
n = f.signals_in_file
signal_labels = f.getSignalLabels()
sigbufs = np.zeros((n, f.getNSamples()[0]))
for i in np.arange(n):
        sigbufs[i, :] = f.readSignal(i)