'''

    Author Nuwan T. Attygalle
    Simple NN implementation with custom loss and activation

'''
# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten
from tensorflow.keras import Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import learning_curve, train_test_split
from tensorflow import keras

import numpy as np


from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, silhouette_samples, silhouette_score
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint

from layers.CustomDense import CustomDense

# %%
# load iris dataset
data = load_iris(as_frame=True)
df = pd.DataFrame(data['data'])
df['label'] = data['target']

# shuffling again for optimum performance ( not mandatory )
SHUFFLE_BUFFER_SIZE = 100000 # (set to a larger value or length of the dataset) len(trainData)
BATCH_SIZE = 10
LEARNING_RATE = 0.0002
PATIENCE = 50
PATIENCE = 50
EPOCHS = 100
VERBOSITY = 1

alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218]
beta_initializer = [2.383, 0.0, 1.0]
# %%
# implement a simple dense network for classifying.
def getModel():

    # test model
    model = Sequential()
    model.add(Input(shape=(4)))
    model.add(Dense(4, activation="LeakyReLU"))
    # model.add(CustomDense(alpha_initializer, beta_initializer, shared_axes=[1]))
    model.add(Dense(32, activation='LeakyReLU'))
    model.add(Dense(3, activation='softmax'))

    # this architecture gives 100% accuracy on the test set
    # model = Sequential()
    # model.add(Input(shape=(4)))
    # model.add(Dense(4, activation="LeakyReLU"))
    # model.add(Dense(32, activation='LeakyReLU'))
    # model.add(Dense(64, activation='LeakyReLU'))
    # model.add(Dense(128, activation='LeakyReLU'))
    # model.add(Dense(64, activation='LeakyReLU'))
    # model.add(Dense(32, activation='LeakyReLU'))
    # model.add(Dense(3, activation='softmax'))

    return model

def CustomModel():
    inputs = Input(shape=(4,))
    layer1 = Dense(4, activation="LeakyReLU")(inputs)
    layer2 = CustomDense(alpha_initializer, beta_initializer, shared_axes=[1])(layer1)
    layer3 = Dense(32, activation="LeakyReLU")(layer2)
    outputs = Dense(3)(layer3) 
    # outputs = Flatten()(outputs)

    # outputs = Reshape((1,3))(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


# %%
# model = getModel()
model = CustomModel()
# %%
model.summary()
# %%
# split the dataset into partitions
targets = to_categorical(df['label'].values).astype(np.int32)

# %%
x_train, x, y_train, y = train_test_split(df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].values, targets, test_size=0.3, random_state=42)
x_test, x_val, y_test, y_val = train_test_split(x, y, test_size=0.3, random_state=42)
# %%
# converting keras dataset into tensors ( not mandatory )
trainData = tf.data.Dataset.from_tensor_slices((x_train, y_train))
valData = tf.data.Dataset.from_tensor_slices((x_val, y_val))

trainData = trainData.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
valData = valData.batch(BATCH_SIZE)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
print(x_val.shape, y_val.shape)
# %% 
optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
# Instantiate a loss function.
# loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

def custom_loss(real, generated):
#   real_loss = loss_obj(tf.ones_like(real), real)

#   generated_loss = loss_obj(tf.zeros_like(generated), generated)

#   total_disc_loss = real_loss + generated_loss

#   return total_disc_loss * 0.5
    return loss_fn(real, generated)

@tf.function
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:

        # Run the forward pass of the layer.
        # The operations that the layer applies
        # to its inputs are going to be recorded
        # on the GradientTape.
        logits = model(x_batch, training=True)  # Logits for this minibatch

        # Compute the loss value for this minibatch.
        # loss_value = loss_fn(y_batch, logits) 
        loss_value = custom_loss(y_batch, logits)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss_value, model.trainable_weights)

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return loss_value

for epoch in range(EPOCHS):
    print("\nStart epoch", epoch)

    for step, (x_train_batch, y_train_batch) in enumerate(trainData):
        # Train the discriminator & generator on one batch of real images.
        loss = train_step(x_train_batch, y_train_batch)

        # Logging.
        if step % 200 == 0:
            # Print metrics
            # print("discriminator loss at step %d: %.2f" % (step, d_loss))
            print("loss at step %d: %.2f" % (step, loss))

            # Save one generated image
        # To limit execution time we stop after 10 steps.
        # Remove the lines below to actually train the model!
        if step > 10:
            break
# # %%
# # --- if you want to use inbluilt fit for training. 
# # defining some middlewares
# # middleware for loging training, validation losses and accuracies
# visualizer = tf.keras.callbacks.TensorBoard(log_dir='./tf_events')
# # middleware for early stopping (stop if the loss doens't improve ) and saving the best model
# earlystops = tf.keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True, verbose=1)#, monitor='val_acc', mode='max')
# filepath = './models/model.h5' # path of the model
# # saving the checkpionts so we can restore the training later
# checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, \
#                         save_best_only=True, save_weights_only=False, \
#                         mode='auto', save_frequency=1)

# # compiling the model
# model.compile(loss=tf.keras.losses.categorical_crossentropy,
#                 optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
#                 metrics=['accuracy'])

# # running the model
# history = model.fit(trainData,
#             batch_size=BATCH_SIZE,
#             epochs=EPOCHS,
#             verbose=VERBOSITY,
#             validation_data=valData,
#             callbacks=[visualizer, earlystops, checkpoint])

# # --- end
# # %%
# # --- your own training loop
# # Instantiate an optimizer.
# optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
# # Instantiate a loss function.
# # loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
# for epoch in range(EPOCHS):
#     print("\nStart of epoch %d" % (epoch,))

#     # Iterate over the batches of the dataset.
#     for step, (x_batch_train, y_batch_train) in enumerate(trainData):

#         # Open a GradientTape to record the operations run
#         # during the forward pass, which enables auto-differentiation.
#         with tf.GradientTape() as tape:

#             # Run the forward pass of the layer.
#             # The operations that the layer applies
#             # to its inputs are going to be recorded
#             # on the GradientTape.
#             logits = model(x_batch_train, training=True)  # Logits for this minibatch

#             # Compute the loss value for this minibatch.
#             loss_value = loss_fn(y_batch_train, logits)

#         # Use the gradient tape to automatically retrieve
#         # the gradients of the trainable variables with respect to the loss.
#         grads = tape.gradient(loss_value, model.trainable_weights)

#         # Run one step of gradient descent by updating
#         # the value of the variables to minimize the loss.
#         optimizer.apply_gradients(zip(grads, model.trainable_weights))

#         # Log every 200 batches.
#         if step % 200 == 0:
#             print(
#                 "Training loss (for one batch) at step %d: %.4f"
#                 % (step, float(loss_value))
#             )
#             print("Seen so far: %s samples" % ((step + 1) * BATCH_SIZE))
# # %%
# evaluating the model
# results = model.evaluate(x_test, y_test)
# %%
# reporting auc and acc for test set
pred = model.predict(x_test)
pred_y = to_categorical(pred.argmax(1), num_classes=3).astype(np.int32)
auc = 100*roc_auc_score(y_test, pred_y , average='weighted', multi_class='ovo')
acc = 100*accuracy_score(y_test, pred_y)
print('Test accuracy: {:.5f}, AUC {:.5f}\n'.format( acc, auc))
# %%
# model.save('models/model_acc_100.h5')
# %%
