import tensorflow as tf
from keras import layers
from keras.layers import Dense, Embedding, LayerNormalization, MultiHeadAttention, Dropout, Add

# Hyperparameters (remain constant through whole model)
dropout_rate = 0.4  # Dropout rate
Dh = 64  # dimension of weights (MSA)
k = 16  # num of heads in MSA

# !!!!!!!! The original ViT paper (and Attention is all you need) suggest Dh to always be equal to De/k !!!!!!!!!!!!!!!
# And here they don't apply that rule !!!!!!


# Electrode Patch encoder
class LinearEmbedding(layers.Layer):
    def __init__(self, num_patches, projection_dim, expand=True):
        super(LinearEmbedding, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.expand = expand
        # Create class token
        w_init = tf.random_normal_initializer()
        class_token = w_init(shape=(1, projection_dim), dtype="float32")
        self.class_token = tf.Variable(initial_value=class_token, trainable=True)
        # Dense layer for linear transformation of electrode patches (Map to constant size De)
        self.projection = Dense(projection_dim)
        # Embedding layer for positional embeddings
        self.position_embedding = Embedding(input_dim=num_patches + 1, output_dim=projection_dim)
        self.dropout = Dropout(0.1)

    def call(self, patch, *kwargs):
        if self.expand is True:  # For electrode-level spatial learning
            # expand dimension 1, so that we can stack the transformer outputs in the brain-region-level
            patch = tf.expand_dims(patch, axis=1)
            # get batch_size (must use tf.shape cause batch_size varies since is it not perfectly divisible)
            batch = tf.shape(patch)[0]
            # augment class token's first dimension to match the batch_size
            class_token = tf.tile(self.class_token, multiples=[batch, 1])
            # reshape the class token to match patches dimensions
            # from (batch,De) to (batch,1,1,De)
            class_token = tf.reshape(class_token, (batch, 1, 1, self.projection_dim))
            # calculate patch embeddings
            patches_embed = self.projection(patch)
            # shape: (None, 1, N, De)
            patches_embed = tf.concat([class_token, patches_embed], 2)
            # calculate position embeddings
            positions = tf.range(start=0, limit=self.num_patches + 1, delta=1)
            positions_embed = self.position_embedding(positions)
            # Add positions to patches
            encoded = patches_embed + positions_embed

        else:  # For brain-region-level spatial learning
            # we do the same as before,but we don't expand dimensions;it's already stacked (concat) on top of each other
            batch = tf.shape(patch)[0]
            class_token = tf.tile(self.class_token, multiples=[batch, 1])
            class_token = tf.reshape(class_token, (batch, 1, self.projection_dim))
            patches_embed = self.projection(patch)
            patches_embed = self.dropout(patches_embed)
            patches_embed = tf.concat([class_token, patches_embed], 1)
            # calculate position embeddings
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
class TransformerEncoderBlock(layers.Layer):
    def __init__(self, model_dim, num_heads=k, msa_dimensions=Dh):
        super(TransformerEncoderBlock, self).__init__()
        self.model_dim = model_dim
        self.layernormalization1 = LayerNormalization(epsilon=1e-6)
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=msa_dimensions, dropout=dropout_rate)
        self.layernormalization2 = LayerNormalization(epsilon=1e-6)
        self.mlp = MLP(hidden_states=model_dim * 4, output_states=model_dim)

    def call(self, x, *kwargs):
        # layer normalization 1.
        x1 = self.layernormalization1(x)  # encoded_patches
        # create a multi-head attention layer.
        attention_output = self.attention(x1, x1)
        # skip connection 1.
        x2 = Add()([attention_output, x])  # encoded_patches
        # layer normalization 2.
        x3 = self.layernormalization2(x2)
        # mLP.
        x3 = self.mlp(x3)
        # skip connection 2.
        y = Add()([x3, x2])
        return y


#  Transformer Encoder Block x L Repeat
class TransformerEncoder(layers.Layer):
    def __init__(self, model_dim, num_blocks):
        super(TransformerEncoder, self).__init__()
        self.blocks = [TransformerEncoderBlock(model_dim, num_blocks) for _ in range(num_blocks)]

    def call(self, x, *kwargs):
        # create a [batch_size, projection_dim] tensor.
        for block in self.blocks:
            x = block(x)
        return x


