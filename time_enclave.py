from tensorflow.keras.models import load_model, Sequential
import tensorflow as tf

import numpy as np
import time
import json
import sys
import os

import experiment_utils

import ennclave_inference 
import mit.prepare_data as mit_prepare_data
import amazon.prepare_data as amazon_prepare_data
import flowers.prepare_data as flowers_prepare_data
import mnist.prepare_data as mnist_prepare_data

def time_from_file(model_path, samples, num_classes, has_enclave):
    model = load_model(model_path)
    time_dict = time_enclave_prediction(model, samples, num_classes, has_enclave)
    return time_dict

def _predict_samples(samples, num_classes, forward):
    #  print("\n\nPredicting with " + str(forward))
    # do the work of the enclave layer by hand to make CPU timing easier
    result = np.zeros((samples.shape[0], num_classes))
    print("Predicting")
    for i, x in enumerate(samples):
        return_bytes = forward(x.astype(np.float32).tobytes(), np.prod(x.shape), num_classes)
        return_values = np.frombuffer(return_bytes, dtype=np.float32)
        label = np.argmax(return_values)
        #  print('label: %d\n' % label)

        if label >= num_classes or label < 0:
            print("Got label %d, out of %d possible..." % (label, num_classes))
            continue

        result[i] = return_values
    return result
    
def time_enclave_prediction(model, samples, num_classes, has_enclave):
    if has_enclave:
        print("\n\nMeasuring enclave\n\n")

        before = time.time()
        # predict dataset
        tf_prediction = model(samples)
        after_tf = time.time()

        tf_prediction = tf_prediction.numpy()

        # final_prediction = enclave_part(tf_prediction)
        # before_setup = time.time()
        # ennclave_inference.initialize()
        # after_setup = time.time()


        enclave_results = _predict_samples(tf_prediction, num_classes, ennclave_inference.sgx_forward)
        after_enclave = time.time()

        # pymatutil.teardown()
        # after_teardown = time.time()

        before_native = time.time()
        native_results = _predict_samples(tf_prediction, num_classes, ennclave_inference.native_forward)
        after_native = time.time()

        enclave_label = np.argmax(enclave_results, axis=1)
        enclave_label = int(enclave_label[0]) # numpy does some type stuff we have to fix
        native_label = np.argmax(native_results, axis=1)
        native_label = int(native_label[0])
            
        print('\n')
        print('Enclave label: %d' % enclave_label)
        print('Native label: %d' % native_label)

    else:
        print("\n\nNOT Measuring enclave\n\n")
        # there is no enclave
        tf_part = model
        
        before = time.time()
        tf_prediction = tf_part(samples)
        after_tf = time.time()

        # set everything so it doesn't muddy the measurement
        before_setup = after_tf
        after_setup = after_tf
        after_enclave = after_tf
        after_teardown = after_tf
        before_native = after_tf
        after_native = after_tf

        print(f"\nTF label {int(np.argmax(tf_prediction,axis=1)[0])}")

        enclave_label = -1
        native_label = -1



    tf_time = after_tf - before
    # enclave_setup_time = after_setup - before_setup
    enclave_time = after_enclave - after_tf
    # teardown_time = after_teardown - after_enclave
    native_time = after_native - before_native

    time_dict = {
        # 'enclave_setup_time': enclave_setup_time,
        'tf_time': tf_time,
        'enclave_time': enclave_time,
        # 'teardown_time': teardown_time,
        # 'combined_enclave_time': enclave_time+enclave_setup_time+teardown_time,
        'native_time': native_time,
        'enclave_label': enclave_label,
        'native_label': native_label
    }
    
    return time_dict

def _dump_to_csv(time_dict, f, write_header=False):
    if write_header:
        # write header line
        for i,k in enumerate(time_dict):
            if i > 0:
                f.write(',')
            f.write(k)
        f.write('\n')

    for i,kv in enumerate(time_dict.items()):
        k,v = kv
        if i > 0:
            f.write(',')
        f.write(str(v))
    f.write('\n')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage {} path/to/enclave_model layers_in_enclave".format(sys.argv[0]))
        sys.exit(1)

    model_path = sys.argv[1]
    layers_in_enclave = int(sys.argv[2])
    dataset = experiment_utils.get_dataset_from_model_path(model_path)

    np.random.seed(1337)
    sample_index = 42
    if dataset == 'mit':
        x_test, y_test = mit_prepare_data.load_test_set()
        sample_index = 20
        num_classes = 67
    elif dataset == 'mnist':
        x_test, y_test = mnist_prepare_data.load_test_set()
        num_classes = 10
    elif dataset == 'amazon':
        _, _, x_test, y_test = amazon_prepare_data.load_cds(20000, 500)
        num_classes = 5
    elif dataset == 'flowers':
        _, _, x_test, y_test = flowers_prepare_data.load_data()
        num_classes = 5
    else:
        raise ValueError("Unknown dataset " + dataset)

    samples = x_test[sample_index:sample_index+1]
    
    time_dict = time_from_file(model_path, samples, num_classes, has_enclave=layers_in_enclave>0)
    time_dict['layers_in_enclave'] = layers_in_enclave
    time_dict['correct_label'] = int(y_test[sample_index])

    print("\n\n")
    print(json.dumps(time_dict, indent=2))

    output_file = 'timing_logs/%s_times.csv' % dataset
    print("Saving to file {}".format(output_file))
    file_existed = os.path.isfile(output_file)
    with open(output_file, 'a+') as f:
        _dump_to_csv(time_dict, f, write_header = not file_existed)
