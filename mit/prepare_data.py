import json
import os

import numpy as np

import experiment_utils


def generate_class_labels(test_images_path, label_file_path):
    with open(test_images_path, 'r') as label_file:
        all_labels = [label.split('/')[0] for label in label_file]

    labels = set(all_labels)
    label_dict = {}
    i = 0
    for label in labels:
        label_dict[label] = i
        i += 1

    with open(label_file_path, 'w+') as label_file:
        json.dump(label_dict, label_file, indent=2)


def load_test_set(data_dir='datasets', test_file='TestImages.txt'):
    full_path = os.path.join(data_dir, 'mit')

    try:
        print("Trying to load previously generated test data")
        x_test = np.load(os.path.join(full_path, 'x_test.npy'))
        y_test = np.load(os.path.join(full_path, 'y_test.npy'))

    except IOError:
        x_test, y_test = _generate_numpy(full_path, test_file, 'test')

    return x_test, y_test


def load_train_set(data_dir='datasets', train_file='TrainImages.txt'):
    full_path = os.path.join(data_dir, 'mit')

    try:
        print("Trying to load previously generated training data")
        x_train = np.load(os.path.join(full_path, 'x_train.npy'))
        y_train = np.load(os.path.join(full_path, 'y_train.npy'))

    except IOError:
        x_train, y_train = _generate_numpy(full_path, train_file, 'train')

    return x_train, y_train


def load_labels(full_path):
    label_file_path = os.path.join(full_path, "class_labels.json")
    if not os.path.isfile(label_file_path):
        test_image_path = os.path.join(full_path, "TestImages.txt")
        generate_class_labels(test_image_path, label_file_path)
    with open(label_file_path, 'r') as f:
        labels = json.load(f)
    return labels


def _generate_numpy(full_path, selector_file, file_suffix):
    print("Not found, generating...")
    with open(os.path.join(full_path, selector_file), 'r') as f:
        images = [os.path.join(full_path, l.strip()) for l in f.readlines()]
    labels = load_labels(full_path)
    processed_images, test_labels = _process_images(labels, images)
    x = processed_images
    y = np.array(test_labels)
    print("Saving numpy arrays")
    np.save(os.path.join(full_path, f'x_{file_suffix}.npy'), x)
    np.save(os.path.join(full_path, f'y_{file_suffix}.npy'), y)
    return x, y


def _process_images(labels, images):
    processed_labels = [labels[s.split('/')[-2]] for s in images]
    processed_images = experiment_utils.preprocess_vgg(images)
    return processed_images, processed_labels
