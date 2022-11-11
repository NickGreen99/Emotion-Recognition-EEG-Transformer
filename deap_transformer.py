import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense, Embedding

# Unpickling
with open("deap_input", "rb") as fp:
    data = pickle.load(fp)

# Hyperparameters
De = 16
d = 5
num_electrode_patches = 9

# First brain region (Pre-Frontal)
electrode_patch1 = np.asarray(data[0][0][0]).astype('float32')
N = electrode_patch1.shape[0]


# Electrode Patch encoder
class ElectrodePatchEncoder(layers.Layer):
    def __init__(self, num_electrodes, projection_dim):
        super(ElectrodePatchEncoder, self).__init__()
        self.N = num_electrodes
        self.projection = Dense(projection_dim)
        self.position_embedding = Embedding(
            input_dim=num_electrodes, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=N, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


# Model creation with Keras Functional API
inputs = keras.Input(shape=(N, d))
outputs = ElectrodePatchEncoder(N, De)(inputs)
model = keras.Model(inputs=inputs, outputs=outputs, name="ElectrodePatchEncoder")
model.summary()
