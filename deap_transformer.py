import pickle
from deap_transformer_classes import TransformerEncoder, LinearEmbedding
import numpy as np
from keras.utils import plot_model
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import Model, activations
import tensorflow as tf
from keras.layers import Dense, Reshape, Concatenate
from sklearn.model_selection import LeaveOneOut

# Unpickling
with open("deap_hala_x", "rb") as fp:
    x = pickle.load(fp)

with open("deap_hala_y", "rb") as fp:
    y = pickle.load(fp)

# Shapes: subjects x brain region x sample x electrodes x frequency bands

# Hyperparameters
d = 5  # dimension of a single electrode

De = 8  # input embedding dimension (electrode level)
Dr = 16  # embedding dimension (brain - region level)

Le = 2  # no of encoders (electrode level)
Lr = 2  # no of encoder (brain - region level)

# Model


def HierarchicalTransformer():
    ### Electrode-Level Spatial Learning

    # 1. Brain region (Pre-Frontal)
    N = 4

    electrode_patch_pf = keras.Input(shape=(N, d))
    patch_embeddings = LinearEmbedding(N, De)(electrode_patch_pf)
    output_pf = TransformerEncoder(De, Le)(patch_embeddings)
    output_pf = Reshape((1, De, (N + 1)))(output_pf)  # reshape output from (batch,(N+1),De) to (batch,(N+1),De)
    output_pf = Dense(4)(output_pf)
    output_pf = Reshape((1, 4, De))(output_pf)

    # 2. Brain region (Frontal)
    N = 5

    electrode_patch_f = keras.Input(shape=(N, d))
    patch_embeddings = LinearEmbedding(N, De)(electrode_patch_f)
    output_f = TransformerEncoder(De, Le)(patch_embeddings)
    output_f = Reshape((1, De, (N + 1)))(output_f)  # reshape output from (batch,(N+1),De) to (batch,(N+1),De)
    output_f = Dense(4)(output_f)
    output_f = Reshape((1, 4, De))(output_f)

    # 3. Brain region (Left Temporal)
    N = 3

    electrode_patch_lt = keras.Input(shape=(N, d))
    patch_embeddings = LinearEmbedding(N, De)(electrode_patch_lt)
    output_lt = TransformerEncoder(De, Le)(patch_embeddings)

    # 4. Brain region (Central)
    N = 5

    electrode_patch_c = keras.Input(shape=(N, d))
    patch_embeddings = LinearEmbedding(N, De)(electrode_patch_c)
    output_c = TransformerEncoder(De, Le)(patch_embeddings)
    output_c = Reshape((1, De, (N + 1)))(output_c)  # reshape output from (batch,(N+1),De) to (batch,(N+1),De)
    output_c = Dense(4)(output_c)
    output_c = Reshape((1, 4, De))(output_c)

    # 5. Brain region (Right Temporal)
    N = 3

    electrode_patch_rt = keras.Input(shape=(N, d))
    patch_embeddings = LinearEmbedding(N, De)(electrode_patch_rt)
    output_rt = TransformerEncoder(De, Le)(patch_embeddings)

    # 6. Brain region (Left Parietal)
    N = 3

    electrode_patch_lp = keras.Input(shape=(N, d))
    patch_embeddings = LinearEmbedding(N, De)(electrode_patch_lp)
    output_lp = TransformerEncoder(De, Le)(patch_embeddings)

    # 7. Brain region (Parietal)
    N = 3

    electrode_patch_p = keras.Input(shape=(N, d))
    patch_embeddings = LinearEmbedding(N, De)(electrode_patch_p)
    output_p = TransformerEncoder(De, Le)(patch_embeddings)

    # 8. Brain region (Right Parietal)
    N = 3

    electrode_patch_rp = keras.Input(shape=(N, d))
    patch_embeddings = LinearEmbedding(N, De)(electrode_patch_rp)
    output_rp = TransformerEncoder(De, Le)(patch_embeddings)

    # 9. Brain region (Occipital)
    N = 3

    electrode_patch_o = keras.Input(shape=(N, d))
    patch_embeddings = LinearEmbedding(N, De)(electrode_patch_o)
    output_o = TransformerEncoder(De, Le)(patch_embeddings)

    # Latent features obtained by the 9 transformers
    xl = Concatenate(axis=1)([output_pf, output_f, output_lt, output_c, output_rt,
                              output_lp, output_p, output_rp, output_o])

    ### Brain-Region-Level Spatial Learning
    brain_regions_N = xl.shape[1]

    # Reshape latent features tensor from (9, 4, 8) to (9, 32)
    xl = Reshape((brain_regions_N, 4 * De))(xl)
    brain_regions_embeddings = LinearEmbedding(brain_regions_N, Dr, False)(xl)
    outputs_br = TransformerEncoder(Dr, Lr)(brain_regions_embeddings)
    class_token_output = outputs_br[:, 0, :]
    # Only class token is input to our emotion prediction NN
    prediction = Dense(2, activation=activations.sigmoid)(class_token_output)

    model = Model(inputs=[electrode_patch_pf, electrode_patch_f, electrode_patch_lt, electrode_patch_c,
                          electrode_patch_rt, electrode_patch_lp, electrode_patch_p, electrode_patch_rp,
                          electrode_patch_o],
                  outputs=prediction)

    return model


model = HierarchicalTransformer()
'''
model.summary()
plot_model(model, to_file='model.png', show_shapes=True)
'''


average = []

callback = tf.keras.callbacks.EarlyStopping(monitor='loss')
model.compile(
    loss=keras.losses.binary_crossentropy,
    optimizer=keras.optimizers.Adam(learning_rate=0.1),
    metrics=["accuracy"]
)

# Leave-One-Subject-Out
loo = LeaveOneOut()

for train_index, test_index in loo.split(x):
    for i in train_index:
        train_data = x[i]
        label_data = y[i]

        history = model.fit(
            x=[train_data[0], train_data[1], train_data[2], train_data[3], train_data[4], train_data[5],
               train_data[6], train_data[7], train_data[8]],
            y=label_data,
            epochs=80, batch_size=512#, callbacks=[callback]
        )
        break
    break
'''
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
'''