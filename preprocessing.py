import pickle

# Unpickling
with open("data", "rb") as fp:
    data = pickle.load(fp)

with open("data_headers", "rb") as fp2:
    data_headers = pickle.load(fp2)
