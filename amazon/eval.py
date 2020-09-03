import argparse

import numpy as np
from tensorflow.keras.models import load_model

from amazon.prepare_data import load_books, load_cds


def eval_true_accuracy(model, x_train, y_train, x_test, y_test):
    print("Generating true training accuracy")
    train_acc, train_mae = eval_predictions(model, x_train, y_train)
    print(f'True training accuracy: {train_acc * 100:.4}')
    print(f'Training MAE: {train_mae:.4}')

    print("Generating true test accuracy")
    test_acc, test_mae = eval_predictions(model, x_test, y_test)

    print(f'True test accuracy: {test_acc * 100:.4}')
    print(f'Test MAE: {test_mae:.4}')


def eval_predictions(model, x, y):
    train_predictions = model.predict(x, verbose=0).flatten()
    train_cleaned_predictions = train_predictions.round()
    train_acc = np.mean(train_cleaned_predictions == y)
    train_errors = np.abs(train_predictions - y)
    train_mae = np.mean(train_errors)
    return train_acc, train_mae


def main():
    parser = argparse.ArgumentParser(description='Evaluate amazon model')
    parser.add_argument('model', type=str, help='path to the model')
    parser.add_argument('dataset', type=str, help='dataset to evaluate on, one of: {books, cds}')
    parser.add_argument('--num-words -n', type=int, dest='num_words', help='Number of words, defaults to 20.000',
                        default=20000)
    parser.add_argument('--sequence-length -s', type=int, dest='sequence_len', help='Sequence length, defaults to 500',
                        default=500)

    args = parser.parse_args()

    model = load_model(args.model)
    dataset = args.dataset

    if dataset == 'books':
        x_train, y_train, x_test, y_test = load_books(args.num_words, args.sequence_len)
    elif dataset == 'cds':
        x_train, y_train, x_test, y_test = load_cds(args.num_words, args.sequence_len)
    else:
        print('ERROR: invalid dataset chosen')
        return

    eval_true_accuracy(model, x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    main()
