import pickle

# Unpickling
with open("data", "rb") as fp:
    data = pickle.load(fp)

with open("data_headers", "rb") as fp2:
    data_headers = pickle.load(fp2)

with open("session_data", "rb") as fp3:
    session = pickle.load(fp3)

# Maybe we need to 1)downsample 2)BP Filter 3)EOG clean

