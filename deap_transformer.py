import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Dense, Embedding, LayerNormalization, MultiHeadAttention, Dropout, Add

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

# !!!!!!!! The original ViT paper (and Attention is all you need) suggest Dh to always be equal to De/k !!!!!!!!!!!!!!!
# And here they don't apply that rule !!!!!!

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
        # Dense layer for linear transformation of electrode patches (Map to constant size De)
        self.projection = Dense(projection_dim)
        # Embedding layer for positional embeddings
        self.position_embedding = Embedding(input_dim=num_patches+1, output_dim=projection_dim)

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
        positions = tf.range(start=0, limit=self.num_patches+1, delta=1)
        positions_embed = self.position_embedding(positions)
        # Add positions to patches
        encoded = patches_embed + positions_embed
        return encoded


# MLP
class MLP(layers.Layer):
    def __init__(self, hidden_states, output_states, dropout = dropout_rate):
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
        self.mlp = MLP(hidden_states=model_dim*2, output_states=model_dim)

    def call(self, x,  *kwargs):
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
        #self.norm = LayerNormalization()
        #self.dropout = Dropout(0.5)

    def call(self, x, *kwargs):
        # Create a [batch_size, projection_dim] tensor.
        for block in self.blocks:
            x = block(x)
        #x = self.norm(x)
        #y = self.dropout(x)
        return x


# Model creation with Keras Functional API
inputs = keras.Input(shape=(N, d))
patch_embeddings = ElectrodePatchEncoder(N, De)(inputs)
outputs = Electrode_Level_Transformer(De)(patch_embeddings)
model = keras.Model(inputs=inputs, outputs=outputs, name="ElectrodePatchEncoder")
model.summary()