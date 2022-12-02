import pickle
from deap_transformer_classes import Electrode_Level_Transformer, ElectrodePatchEncoder, De, d
import numpy as np
from tensorflow import keras
import tensorflow as tf
from keras.layers import Dense

# Unpickling
with open("deap_input", "rb") as fp:
    data = pickle.load(fp)

### Electrode-Level Spatial Learning

# 1. Brain region (Pre-Frontal)
electrode_patch = np.asarray(data[0][0][0]).astype('float32')
N = electrode_patch.shape[0]

# First model creation with Keras Functional API
patch_inputs1 = keras.Input(shape=(N, d))
patch_embeddings = ElectrodePatchEncoder(N, De)(patch_inputs1)
outputs1 = Electrode_Level_Transformer(De)(patch_embeddings)
outputs1 = tf.reshape(outputs1, [(N + 1), De])  # reshape output from (1,(N+1),De) to ((N+1),De)
outputs1 = Dense(4)(tf.transpose(outputs1))
outputs1 = tf.transpose(outputs1)

# 2. Brain region (Frontal)
electrode_patch = np.asarray(data[0][0][1]).astype('float32')
N = electrode_patch.shape[0]

# Second model creation with Keras Functional API
patch_inputs2 = keras.Input(shape=(N, d))
patch_embeddings = ElectrodePatchEncoder(N, De)(patch_inputs2)
outputs2 = Electrode_Level_Transformer(De)(patch_embeddings)
outputs2 = tf.reshape(outputs2, [(N + 1), De])  # reshape output from (1,(N+1),De) to ((N+1),De)
outputs2 = Dense(4)(tf.transpose(outputs2))
outputs2 = tf.transpose(outputs2)

# 3. Brain region (Left Temporal)
electrode_patch = np.asarray(data[0][0][2]).astype('float32')
N = electrode_patch.shape[0]

# Third model creation with Keras Functional API
patch_inputs3 = keras.Input(shape=(N, d))
patch_embeddings = ElectrodePatchEncoder(N, De)(patch_inputs3)
outputs3 = Electrode_Level_Transformer(De)(patch_embeddings)
outputs3 = tf.reshape(outputs3, [(N + 1), De])  # reshape output from (1,(N+1),De) to ((N+1),De)

# 4. Brain region (Central)
electrode_patch = np.asarray(data[0][0][3]).astype('float32')
N = electrode_patch.shape[0]

# Fourth model creation with Keras Functional API
patch_inputs4 = keras.Input(shape=(N, d))
patch_embeddings = ElectrodePatchEncoder(N, De)(patch_inputs4)
outputs4 = Electrode_Level_Transformer(De)(patch_embeddings)
outputs4 = tf.reshape(outputs4, [(N + 1), De])  # reshape output from (1,(N+1),De) to ((N+1),De)
outputs4 = Dense(4)(tf.transpose(outputs4))
outputs4 = tf.transpose(outputs4)

# 5. Brain region (Right Temporal)
electrode_patch = np.asarray(data[0][0][4]).astype('float32')
N = electrode_patch.shape[0]

# Fifth model creation with Keras Functional API
patch_inputs5 = keras.Input(shape=(N, d))
patch_embeddings = ElectrodePatchEncoder(N, De)(patch_inputs5)
outputs5 = Electrode_Level_Transformer(De)(patch_embeddings)
outputs5 = tf.reshape(outputs5, [(N + 1), De])  # reshape output from (1,(N+1),De) to ((N+1),De)

# 6. Brain region (Left Parietal)
electrode_patch = np.asarray(data[0][0][5]).astype('float32')
N = electrode_patch.shape[0]

# Sixth model creation with Keras Functional API
patch_inputs6 = keras.Input(shape=(N, d))
patch_embeddings = ElectrodePatchEncoder(N, De)(patch_inputs6)
outputs6 = Electrode_Level_Transformer(De)(patch_embeddings)
outputs6 = tf.reshape(outputs6, [(N + 1), De])  # reshape output from (1,(N+1),De) to ((N+1),De)

# 7. Brain region (Parietal)
electrode_patch = np.asarray(data[0][0][6]).astype('float32')
N = electrode_patch.shape[0]

# Seventh model creation with Keras Functional API
patch_inputs7 = keras.Input(shape=(N, d))
patch_embeddings = ElectrodePatchEncoder(N, De)(patch_inputs7)
outputs7 = Electrode_Level_Transformer(De)(patch_embeddings)
outputs7 = tf.reshape(outputs7, [(N + 1), De])  # reshape output from (1,(N+1),De) to ((N+1),De)

# 8. Brain region (Right Parietal)
electrode_patch = np.asarray(data[0][0][7]).astype('float32')
N = electrode_patch.shape[0]

# Eighth model creation with Keras Functional API
patch_inputs8 = keras.Input(shape=(N, d))
patch_embeddings = ElectrodePatchEncoder(N, De)(patch_inputs8)
outputs8 = Electrode_Level_Transformer(De)(patch_embeddings)
outputs8 = tf.reshape(outputs8, [(N + 1), De])  # reshape output from (1,(N+1),De) to ((N+1),De)

# 9. Brain region (Occipital)
electrode_patch = np.asarray(data[0][0][8]).astype('float32')
N = electrode_patch.shape[0]

# Ninth model creation with Keras Functional API
patch_inputs9 = keras.Input(shape=(N, d))
patch_embeddings = ElectrodePatchEncoder(N, De)(patch_inputs9)
outputs9 = Electrode_Level_Transformer(De)(patch_embeddings)
outputs9 = tf.reshape(outputs9, [(N + 1), De])  # reshape output from (1,(N+1),De) to ((N+1),De)

# Latent features obtained by the 9 transformers
xl = tf.stack([outputs1, outputs2, outputs3, outputs4, outputs5, outputs6, outputs7, outputs8, outputs9])

### Brain-Region-Level Spatial Learning
brain_regions_N = xl.shape[0]
