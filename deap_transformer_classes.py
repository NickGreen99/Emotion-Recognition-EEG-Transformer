import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models
from keras.layers import Dense, Embedding, LayerNormalization, MultiHeadAttention, Dropout, Add, concatenate

# Unpickling
with open("deap_input", "rb") as fp:
    data = pickle.load(fp)

regions = len(data[0][0])

# Hyperparameters
d = 5  # dimension of a single electrode
num_electrode_patches = 9  # number of electrode patches/brain regions/transformers

De = 8  # input embedding dimension (electrode level)
Dr = 16  # embedding dimension (brain - region level)
Dh = 64  # dimension of weights (MSA)
k = 16  # num of heads in MSA

# !!!!!!!! The original ViT paper (and Attention is all you need) suggest Dh to always be equal to De/k !!!!!!!!!!!!!!!
# And here they don't apply that rule !!!!!!

Le = 2  # no of encoders (electrode level)
Lr = 2  # no of encoder (brain - region level)

dropout_rate = 0.4  # Dropout rate


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
        # Dense layer for linear transformation of electrode patches (Map to constant size De)
        self.projection = Dense(projection_dim)
        # Embedding layer for positional embeddings
        self.position_embedding = Embedding(input_dim=num_patches + 1, output_dim=projection_dim)

    def call(self, patch, *kwargs):
        # Reshape the class token to match patches dimensions
        # From (1,16) to (1,1,16)
        class_token = tf.reshape(self.class_token, (1, 1, self.projection_dim))
        # Calculate patch embeddings
        patches_embed = self.projection(patch)
        # Shape: (None, 4, 16)
        patches_embed = tf.concat([patches_embed, class_token], 1)
        # Shape (1 ,5, 16) -- note: in concat all dimensions EXCEPT axis must be equal
        # Calculate position embeddings
        positions = tf.range(start=0, limit=self.num_patches + 1, delta=1)
        positions_embed = self.position_embedding(positions)
        # Add positions to patches
        encoded = patches_embed + positions_embed
        return encoded


# MLP
class MLP(layers.Layer):
    def __init__(self, hidden_states, output_states, dropout=dropout_rate):
        super(MLP, self).__init__()
        self.dense1 = Dense(hidden_states, activation=tf.nn.gelu)
        self.dense2 = Dense(output_states, activation=tf.nn.gelu)
        self.dropout = Dropout(dropout)

    def call(self, x, *kwargs):
        hidden = self.dense1(x)
        dr_hidden = self.dropout(hidden)
        output = self.dense2(dr_hidden)
        dr_output = self.dropout(output)
        return dr_output


# Transformer Encoder Block
class Transformer_Encoder_Block(layers.Layer):
    def __init__(self, model_dim, num_heads=k, msa_dimensions=Dh):
        super(Transformer_Encoder_Block, self).__init__()
        self.model_dim = model_dim
        self.layernormalization1 = LayerNormalization()
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=msa_dimensions, dropout=dropout_rate)
        self.layernormalization2 = LayerNormalization()
        self.mlp = MLP(hidden_states=model_dim * 2, output_states=model_dim)

    def call(self, x, *kwargs):
        # Layer normalization 1.
        x1 = self.layernormalization1(x)  # encoded_patches
        # Create a multi-head attention layer.
        attention_output = self.attention(x1, x1)
        # Skip connection 1.
        x2 = Add()([attention_output, x])  # encoded_patches
        # Layer normalization 2.
        x3 = self.layernormalization2(x2)
        # MLP.
        x3 = self.mlp(x3)
        # Skip connection 2.
        y = Add()([x3, x2])
        return y


#  Transformer Encoder Block x 10 Repeat
class Electrode_Level_Transformer(layers.Layer):
    def __init__(self, model_dim, num_blocks=Le):
        super(Electrode_Level_Transformer, self).__init__()
        self.blocks = [Transformer_Encoder_Block(model_dim) for _ in range(num_blocks)]
        # self.norm = LayerNormalization()
        # self.dropout = Dropout(0.5)

    def call(self, x, *kwargs):
        # Create a [batch_size, projection_dim] tensor.
        for block in self.blocks:
            x = block(x)
        # x = self.norm(x)
        # y = self.dropout(x)
        return x


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

patch_embeddings = ElectrodePatchEncoder(brain_regions_N, De)(patch_inputs8)
outputs8 = Electrode_Level_Transformer(De)(patch_embeddings)
# model = keras.Model(inputs=[patch_inputs1, patch_inputs2, patch_inputs3, patch_inputs4, patch_inputs5, patch_inputs6,
#                             patch_inputs7, patch_inputs8, patch_inputs9], outputs=[xl])
