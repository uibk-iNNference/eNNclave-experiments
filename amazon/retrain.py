import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, clone_model
from tensorflow.keras.optimizers import Adam, SGD

from amazon.eval import eval_true_accuracy
from amazon.prepare_data import load_cds, rebuild_cds

SEED = 1337
tf.random.set_seed(SEED)
np.random.seed(SEED)

MODEL_FILE = 'models/amazon.h5'

NUM_WORDS = 20000
SEQUENCE_LENGTH = 500

DROPOUT_RATE = 0.3
HIDDEN_NEURONS = 600
EPOCHS = 15  # this is where we start to overfit
RETRAIN_EPOCHS = 1


def prepare_model(num_layers, original_trainable=False):
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    source_model = load_model(MODEL_FILE)

    ret_model = Sequential()
    for layer in source_model.layers[:-num_layers]:
        layer.trainable = original_trainable
        ret_model.add(layer)

    for layer in source_model.layers[-num_layers:]:
        layer.trainable = True
        ret_model.add(layer)

    return ret_model


def train(model, x, y, epochs, retrain=False):
    if retrain:
        optimizer = SGD(lr=0.0001)
    else:
        optimizer = Adam()
    model.compile(loss='mean_absolute_error', optimizer=optimizer, metrics=['mae', 'acc'])
    hist = model.fit(
        x,
        y,
        epochs=epochs,
        shuffle=True,
        verbose=0,
    )
    return hist


def eval_split(num_layers):
    fixed_model = prepare_model(num_layers, False)
    print(f"training for {EPOCHS} epochs with extractor fixed")
    train(fixed_model, x_train, y_train, EPOCHS)
    eval_true_accuracy(fixed_model, x_train, y_train, x_test, y_test)
    fixed_model.save(f'models/amazon_{num_layers}_fixed.h5')

    print()
    flexible_model = prepare_model(num_layers, True)
    print(f"training for {EPOCHS} epochs with extractor flexible")
    train(flexible_model, x_train, y_train, EPOCHS)
    eval_true_accuracy(flexible_model, x_train, y_train, x_test, y_test)
    flexible_model.save(f'models/amazon_{num_layers}_flexible.h5')


def main():
    x_train, y_train, x_test, y_test = load_cds(NUM_WORDS, SEQUENCE_LENGTH, seed=SEED)

    # get original accuracy
    original_model = load_model(MODEL_FILE)
    print("Original model on cd data:")
    eval_true_accuracy(original_model, x_train, y_train, x_test, y_test)

    print("\n\n####### Retraining last layer #######")
    eval_split(1)

    # retrain dense layers
    print("\n\n\n####### Retraining dense layers #######")
    print("####### This was used in paper  #######")
    eval_split(6)

    # retrain conv and dense layers
    print("\n\n####### Keeping only embedding and tokenizer #######")
    eval_split(15)

    #  retrain entire network
    print("\n\n####### Keeping only tokenizer #######")
    eval_split(16)

    #  rebuild even tokenizer
    print("\n\n####### rebuilding everything #######")
    original_model = load_model(MODEL_FILE)
    x_train, y_train, x_test, y_test, _ = rebuild_cds(NUM_WORDS, SEQUENCE_LENGTH, seed=SEED)

    new_model = Sequential()
    for layer in original_model.layers:
        layer.trainable = True
        new_model.add(layer)

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    print(f"training for {EPOCHS} epochs")
    train(new_model, x_train, y_train, EPOCHS)
    eval_true_accuracy(new_model, x_train, y_train, x_test, y_test)
    new_model.save('models/amazon_new.h5')

if __name__ == "__main__":
    main()