from os.path import join
import os

import tensorflow as tf
import tensorflow.keras.applications as apps
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
from numpy.random import seed
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.models import Sequential, load_model

seed(1337)
tf.compat.v1.set_random_seed(1337)

import experiment_utils
import mit.prepare_data as prepare_data

# config
HIDDEN_NEURONS = 2048
DROPOUT_RATIO = 0.4
STEPS_PER_EPOCH = 3


def build_model(extractor: Sequential, num_extractor_layers, trainable=False):
    ret = Sequential()
    for i in range(num_extractor_layers):
        layer = extractor.layers[i]
        layer.trainable = trainable
        ret.add(layer)

    ret.add(layers.GlobalAveragePooling2D())

    ret.add(layers.Dense(HIDDEN_NEURONS, activation='relu'))
    ret.add(layers.Dropout(DROPOUT_RATIO))
    ret.add(layers.Dense(HIDDEN_NEURONS, activation='relu'))
    ret.add(layers.Dropout(DROPOUT_RATIO))
    ret.add(layers.Dense(67, activation='softmax'))

    return ret


def train(model, target_file, train_ds, test_ds, epochs=2000, learning_rate=0.001, model_dir='models'):
    optimizer = optimizers.Adam(learning_rate=learning_rate)

    target_path = join(model_dir, target_file)
    print('Hyperparameters:')
    print('num_epochs: {}'.format(epochs))
    print('hidden_neurons: {}'.format(HIDDEN_NEURONS))
    print()

    model.compile(optimizer=optimizer,
                  loss=sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    model.fit(train_ds,
              epochs=epochs,
              steps_per_epoch=STEPS_PER_EPOCH,
              validation_data=test_ds,
              validation_steps=STEPS_PER_EPOCH,
              verbose=2)

    loss0, accuracy0 = model.evaluate(test_ds)

    print("loss: {:.2f}".format(loss0))
    print("accuracy: {:.2f}".format(accuracy0))
    print("\nSaving model at: {}".format(target_path))
    model.save(target_path)


def main():
    x_train, y_train = prepare_data.load_train_set()
    train_ds = experiment_utils.generate_dataset(
        x_train, y_train, preprocess_function=None)
    del x_train, y_train
    x_test, y_test = prepare_data.load_test_set()
    test_ds = experiment_utils.generate_dataset(
        x_test, y_test, shuffle=False, repeat=False, preprocess_function=None)
    del x_test, y_test

    # build model
    model_dir = 'models'

    source_path = join(model_dir, 'vgg16_places365.h5')

    print("Training model based on Places database")
    places_unrefined_name = 'mit_places_unrefined.h5'
    places_unrefined_path = join(model_dir, places_unrefined_name)

    if not os.path.exists(places_unrefined_path):
        print('Trying to load model from %s' % source_path)
        places_extractor = load_model(source_path)
        print("Training")
        places_unrefined = build_model(places_extractor, 50, trainable=False)
        train(places_unrefined, places_unrefined_name, train_ds, test_ds)
        del(places_unrefined)
    else:
        print("Found existing, not training new")

    print("Retraining fixed")
    places_fixed = load_model(places_unrefined_path)
    train(places_fixed, 'mit_places_fixed.h5', train_ds, test_ds, epochs=200, learning_rate=0.000001)

    print("Retraining flexible")
    places_flexible = load_model(places_unrefined_path)
    for layer in places_flexible.layers:
        layer.trainable = True
    train(places_flexible, 'mit_places_flexible.h5', train_ds, test_ds, epochs=200, learning_rate=0.000001)

    print("Training model based on Imagenet database")
    imagenet_unrefined_name = 'mit_imagenet_unrefined.h5'
    imagenet_unrefined_path = join(model_dir, imagenet_unrefined_name)
    
    if not os.path.exists(imagenet_unrefined_path):
        print("Training")
        imagenet_extractor = apps.VGG16(
            include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        imagenet_unrefined = build_model(imagenet_extractor, 19, trainable=False)
        train(imagenet_unrefined, imagenet_unrefined_name, train_ds, test_ds)
    else:
        print("Found existing, not training new")

    print("Retraining fixed")
    imagenet_fixed = load_model(imagenet_unrefined_path)
    train(imagenet_fixed, 'mit_imagenet_fixed', train_ds, test_ds, epochs=200, learning_rate=0.000001)
    del(imagenet_fixed)

    print("Retraining flexible")
    imagenet_flexible = load_model(imagenet_unrefined_path)
    for layer in imagenet_flexible.layers:
        layer.trainable = True
    train(imagenet_flexible, 'mit_imagenet_flexible.h5', train_ds, test_ds, epochs=200, learning_rate=0.000001)
    del(imagenet_flexible)


if __name__ == "__main__":
    main()
