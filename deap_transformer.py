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

    hslt = Model(inputs=[electrode_patch_pf, electrode_patch_f, electrode_patch_lt, electrode_patch_c,
                         electrode_patch_rt, electrode_patch_lp, electrode_patch_p, electrode_patch_rp,
                         electrode_patch_o],
                 outputs=prediction)

    return hslt


model = HierarchicalTransformer()
'''
model.summary()
plot_model(model, to_file='model.png', show_shapes=True)
'''

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3, verbose=1)
lr_cosine_decay = keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.1, decay_steps=1000)
adam = keras.optimizers.Adam(learning_rate=lr_cosine_decay)

model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=adam, #keras.optimizers.Adam(learning_rate=0.01),
    metrics=["accuracy"]
)

# Leave-One-Subject-Out
loo = LeaveOneOut()
average_acc = []
count = 0
for train_index, test_index in loo.split(x):
    count += 1
    print('-----------------------------------------')
    print('count = ' + str(count))
    tf.keras.backend.clear_session()

    # Train - Validation Split
    train = []
    train_labels = []

    for i in train_index:
        train.append(x[i])
        train_labels.append(np.reshape(y[i], (-1, 1)))
    test = x[test_index[0]]
    test_labels = y[test_index[0]]

    prefrontal_x = [br[0] for br in train]
    prefrontal_x = np.vstack(prefrontal_x)
    prefrontal_test = test[0]

    frontal_x = [br[1] for br in train]
    frontal_x = np.vstack(frontal_x)
    frontal_test = test[1]

    ltemporal_x = [br[2] for br in train]
    ltemporal_x = np.vstack(ltemporal_x)
    ltemporal_test = test[2]

    central_x = [br[3] for br in train]
    central_x = np.vstack(central_x)
    central_test = test[3]

    rtemporal_x = [br[4] for br in train]
    rtemporal_x = np.vstack(rtemporal_x)
    rtemporal_test = test[4]

    lparietal_x = [br[5] for br in train]
    lparietal_x = np.vstack(lparietal_x)
    lparietal_test = test[5]

    parietal_x = [br[6] for br in train]
    parietal_x = np.vstack(parietal_x)
    parietal_test = test[6]

    rparietal_x = [br[7] for br in train]
    rparietal_x = np.vstack(rparietal_x)
    rparietal_test = test[7]

    occipital_x = [br[8] for br in train]
    occipital_x = np.vstack(occipital_x)
    occipital_test = test[8]

    train_labels = np.vstack(train_labels)

    # One-Hot-Encoding
    train_labels = tf.one_hot(train_labels, 2)
    train_labels = np.reshape(train_labels, (-1, 2))

    test_labels = tf.one_hot(test_labels, 2)

    history = model.fit(
        x=[prefrontal_x, frontal_x, ltemporal_x, central_x, rtemporal_x, lparietal_x,
           parietal_x, rparietal_x, occipital_x],
        y=train_labels,
        validation_data=([prefrontal_test, frontal_test, ltemporal_test, central_test, rtemporal_test, lparietal_test,
                          parietal_test, rparietal_test, occipital_test], test_labels),
        epochs=80, batch_size=512, callbacks=[early_stopping]
    )
    average_acc.append(history.history['val_accuracy'][-1])

average_acc = np.array(average_acc)
print(np.mean(average_acc))
print(average_acc)
'''
# Leave-One-Subject-Out
loo = LeaveOneOut()
average_acc = []
big_count = 0
for train_index, test_index in loo.split(x):
    big_count += 1
    count = 0
    print('-----------------------------------------')
    print('big count = ' + str(big_count))
    tf.keras.backend.clear_session()

    for i in train_index:
        prefrontal = np.vstack(x[i])
    for i in train_index:
        train_data = x[i]
        label_data = y[i]

        count += 1
        print('-----------------------------------------')
        print('count = ' + str(count))
        print('big count = ' + str(big_count))
        history = model.fit(
            x=[train_data[0], train_data[1], train_data[2], train_data[3], train_data[4], train_data[5],
               train_data[6], train_data[7], train_data[8]],
            y=label_data,
            validation_data=(x[test_index[0]], y[test_index[0]]),
            epochs=80, batch_size=512 #callbacks=[early_stopping]
        )
    average_acc.append(history.history['val_accuracy'][-1])
    break
average_acc = np.array(average_acc)
print(np.mean(average_acc))
print(average_acc)
'''

# plot losses
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
