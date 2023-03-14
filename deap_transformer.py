import pickle
from deap_transformer_classes import TransformerEncoder, LinearEmbedding
import numpy as np
from keras.utils import plot_model
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from sklearn import metrics
import math
from tensorflow import keras
from keras import Model, activations
import tensorflow as tf
from keras.layers import Dense, Reshape, Concatenate, GlobalAveragePooling1D
from sklearn.model_selection import LeaveOneOut
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from keras import backend as K

# Unpickling
with open("deap_hvlv_x", "rb") as fp:
    x = pickle.load(fp)

with open("deap_hvlv_y", "rb") as fp:
    y = pickle.load(fp)


def count_data(li):
    ones = 0
    zeros = 0
    for i in li:
        zeros = zeros + i.tolist().count(0)
        ones = ones + i.tolist().count(1)
    return zeros, ones


low, high = count_data(y)
print('Low: ' + str(low) + ' | ' + 'High: ' + str(high))


# Check whether we do binary or 4-class classification
classes = 2
if max(y[0] > 1):
    classes = 4
print(classes)
if classes == 2:
    activation = activations.sigmoid
    loss = 'binary_crossentropy'
else:
    activation = activations.softmax
    loss = 'categorical_crossentropy'

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
    if classes == 2:
        prediction = Dense(classes, activation=activation)(class_token_output)
        print('SIGMOID')
    else:
        prediction = Dense(classes, activation=activation)(class_token_output)
        print('SOFTMAX')
    hslt = Model(inputs=[electrode_patch_pf, electrode_patch_f, electrode_patch_lt, electrode_patch_c,
                         electrode_patch_rt, electrode_patch_lp, electrode_patch_p, electrode_patch_rp,
                         electrode_patch_o],
                 outputs=prediction)

    return hslt


model = HierarchicalTransformer()

model.summary()

plot_model(model, to_file='model.png', show_shapes=True)


# Training hyperparameters
epochs = 80
batch_size = 512
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10,
                                               restore_best_weights=True)


class CosineScheduler:
    def __init__(self, max_update, base_lr, final_lr, warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * (1 + math.cos(math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr


#scheduler = CosineScheduler(max_update=40, base_lr=0.1, final_lr=0.0)
#plt.plot(tf.range(epochs), [scheduler(t) for t in range(epochs)])


# Leave-One-Subject-Out
loo = LeaveOneOut()
average_results_acc = []
average_results_f1 = []
average_results_cohen = []
for train_index, test_index in loo.split(x):
    print('-----------------------------------------')
    print('count = ' + str(test_index[0]))
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
    train_labels = tf.one_hot(train_labels, classes)
    train_labels = np.reshape(train_labels, (-1, classes))
    test_labels = tf.one_hot(test_labels, classes)

    #scheduler = CosineScheduler(max_update=80, base_lr=0.01, final_lr=0)
    #lrate = LearningRateScheduler(scheduler)
    #lrate = LearningRateScheduler(cosine_decay)

    num_train_steps = (train_labels.shape[0] // batch_size) * epochs  # total number of batches

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.005,
        decay_steps=num_train_steps
    )
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(
        loss=loss,
        optimizer=opt
    )

    history = model.fit(
        x=[prefrontal_x, frontal_x, ltemporal_x, central_x, rtemporal_x, lparietal_x,
           parietal_x, rparietal_x, occipital_x],
        y=train_labels,
        validation_data=([prefrontal_test, frontal_test, ltemporal_test, central_test, rtemporal_test, lparietal_test,
                          parietal_test, rparietal_test, occipital_test], test_labels),
        epochs=epochs, batch_size=batch_size, callbacks=[early_stopping]
    )
    prediction = model.predict([prefrontal_test, frontal_test, ltemporal_test, central_test, rtemporal_test,
                                lparietal_test, parietal_test, rparietal_test, occipital_test])
    prediction_bool = np.argmax(prediction, axis=1)
    true_bool = np.argmax(test_labels, axis=1)

    # Confusion Matrix
    if classes == 2:
        accuracy = metrics.accuracy_score(true_bool, prediction_bool)
        f1_score = metrics.f1_score(true_bool, prediction_bool)
        cohen = metrics.cohen_kappa_score(true_bool, prediction_bool)
        print(accuracy)
        print(f1_score)
        print(cohen)
        if not np.isnan(accuracy):
            average_results_acc.append(accuracy)
        if not np.isnan(cohen):
            average_results_cohen.append(cohen)
        if not np.isnan(f1_score):
            average_results_f1.append(f1_score)
    else:
        accuracy = metrics.accuracy_score(true_bool, prediction_bool)
        f1_score = metrics.f1_score(true_bool, prediction_bool, average='weighted')
        cohen = metrics.cohen_kappa_score(true_bool, prediction_bool)
        print(accuracy)
        print(f1_score)
        print(cohen)
        if not np.isnan(accuracy):
            average_results_acc.append(accuracy)
        if not np.isnan(cohen):
            average_results_cohen.append(cohen)
        if not np.isnan(f1_score):
            average_results_f1.append(f1_score)


average_results_acc = np.array(average_results_acc)
average_results_f1 = np.array(average_results_f1)
average_results_cohen = np.array(average_results_cohen)
print('Results Acc: ' + str(np.mean(average_results_acc)) + ' | Std: ' + str(np.std(average_results_acc)))
print('Results F1: ' + str(np.mean(average_results_f1)) + ' | Std: ' + str(np.std(average_results_f1)))
print('Results Kappa: ' + str(np.mean(average_results_cohen)) + ' | Std: ' + str(np.std(average_results_cohen)))

'''
# plot losses
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
'''
