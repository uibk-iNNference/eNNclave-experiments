print("Setting seeds")
from numpy.random import seed

seed(1337)
import tensorflow as tf

tf.random.set_seed(1337)
print("Done")

import tensorflow.keras.layers as layers
import tensorflow.keras.applications as apps
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import sparse_categorical_crossentropy

import pandas as pd

import experiment_utils
import mit.prepare_data as prepare_data

x_train, y_train = prepare_data.load_train_set()
train_ds = experiment_utils.generate_dataset(x_train, y_train, preprocess_function=None)
del x_train, y_train
x_test, y_test = prepare_data.load_test_set()
test_ds = experiment_utils.generate_dataset(
    x_test, y_test, shuffle=False, repeat=False, preprocess_function=None)
del x_test, y_test

# build model
MODEL_FILE = 'models/mit.h5'
HIST_FILE = 'hist_mit.csv'
HIDDEN_NEURONS = 2048
DROPOUT_RATIO = 0.4
NUM_EPOCHS = 2000
STEPS_PER_EPOCH = 3

extractor = apps.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

dense = Sequential([
    layers.Dense(HIDDEN_NEURONS, activation='relu'),
    layers.Dropout(DROPOUT_RATIO),
    layers.Dense(HIDDEN_NEURONS, activation='relu'),
    layers.Dropout(DROPOUT_RATIO),
    layers.Dense(67, activation='softmax')
])

model = Sequential()
for layer in extractor.layers:
    layer.trainable = False
    model.add(layer)
model.add(layers.GlobalAveragePooling2D())
# model.add(layers.MaxPooling2D(2))
# model.add(layers.Flatten())
for layer in dense.layers:
    model.add(layer)

print('Hyperparameters:')
print('num_epochs: {}'.format(NUM_EPOCHS))
print('hidden_neurons: {}'.format(HIDDEN_NEURONS))
# print('training set size: {}'.format(len(y_train)))
# print('test set size: {}'.format(len(y_test)))
print()

model.compile(optimizer='adam',
              loss=sparse_categorical_crossentropy,
              metrics=['accuracy'])

history = model.fit(train_ds,
                    epochs=NUM_EPOCHS,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=test_ds,
                    validation_steps=STEPS_PER_EPOCH)

loss0, accuracy0 = model.evaluate(test_ds)

print("loss: {:.2f}".format(loss0))
print("accuracy: {:.2f}".format(accuracy0))
print("\nSaving model at: {}".format(MODEL_FILE))
model.save(MODEL_FILE)

print("Saving history at: {}".format(HIST_FILE))
hist_df = pd.DataFrame(history.history)
with open(HIST_FILE, 'w+') as f:
    hist_df.to_csv(f)
