import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Dense, Embedding

# Unpickling
with open("deap_input", "rb") as fp:
    data = pickle.load(fp)


# Hyperparameters
d = 5  # dimension of a single electrode
num_electrode_patches = 9  # number of electrode patches/brain regions/transformers

De = 8  # input embedding dimension (electrode level)
Dr = 16  # embedding dimension (brain - region level)
Dh = 64  # dimension of weights (MSA)
k = 16  # num of heads in MSA


Le = 2  # no of encoders (electrode level)
Lr = 2  # no of encoder (brain - region level)

dropout_rate = 0.4  # Dropout rate

# First brain region (Pre-Frontal)
electrode_patch1 = np.asarray(data[0][0][0]).astype('float32')
N = electrode_patch1.shape[0]


# Electrode Patch encoder
class ElectrodePatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(ElectrodePatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        # Create class token
        w_init = tf.random_normal_initializer()
        class_token = w_init(shape=(1, projection_dim), dtype="float32")
        self.class_token = tf.Variable(initial_value=class_token, trainable=True)
        # Dense layer for linear transformation of electrode patches
        self.projection = Dense(projection_dim)
        # Embedding layer for positional embeddings
        self.position_embedding = Embedding(input_dim=num_patches+1, output_dim=projection_dim)

    def call(self, patch):
        # Reshape the class token to match patches dimensions
        # From (1,16) to (1,1,16)
        class_token = tf.reshape(self.class_token, (1, 1, self.projection_dim))
        # Calculate patch embeddings
        patches_embed = self.projection(patch)
        # Shape: (None, 4, 16)
        patches_embed = tf.concat([patches_embed, class_token], 1)
        # Shape (1 ,5, 16) -- note: in concat all dimensions EXCEPT axis must be equal
        # Calculate position embeddings
        positions = tf.range(start=0, limit=self.num_patches+1, delta=1)
        positions_embed = self.position_embedding(positions)
        # Add positions to patches
        encoded = patches_embed + positions_embed
        return encoded


# Model creation with Keras Functional API
inputs = keras.Input(shape=(N, d))
outputs = ElectrodePatchEncoder(N, De)(inputs)
model = keras.Model(inputs=inputs, outputs=outputs, name="ElectrodePatchEncoder")
model.summary()


