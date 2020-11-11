from tensorflow.keras.models import load_model, Sequential

from ennclave import Enclave
from experiment_utils import get_all_layers

import argparse
import pathlib
import os
from invoke import Context
import sys


def get_new_filename(model_path):
    model_path = pathlib.Path(model_path)
    target_dir = model_path.parent
    target_basename = model_path.stem
    target_ending = model_path.suffix

    new_filename = target_basename + '_enclave' + target_ending
    target_file = target_dir.joinpath(new_filename)

    return target_file




def compile_enclave(verbose=False):
    context = Context()
    try:
        eNNclave_home = os.environ['ENNCLAVE_HOME']
    except KeyError:
        print("ENNCLAVE_HOME environment variable not set", file=sys.stderr)
        sys.exit(1)

    with context.cd(os.path.join(eNNclave_home, 'build')):
        result = context.run('make backend_sgx', hide=not verbose)
        if verbose:
            print(result.stdout)

        if not result.ok:
            raise OSError(result.stdout)


def build_enclave(model_file, n):
    print('Loading model from %s' % model_file)
    model = load_model(model_file, custom_objects={'Enclave': Enclave})

    # build flattened model structure
    all_layers = get_all_layers(model)
    num_layers = len(all_layers)

    # extract the last n layers
    enclave = Enclave()
    for i in range(num_layers - n, num_layers):
        layer = all_layers[i]
        enclave.add(layer)

    enclave_input_shape = all_layers[-n].input_shape
    enclave.build(input_shape=enclave_input_shape)
    
    # generate parameter file
    enclave.generate_state()
    # build cpp files and config for sgx
    enclave.generate_forward(backend='sgx')
    enclave.generate_config()

    # generate cpp file for native C
    enclave.generate_forward(backend='native')

    # build replacement layer for original model
    enclave_model = Sequential(all_layers[:-n])
    enclave_model.build(enclave_input_shape)
    print("New model:")
    enclave_model.summary()

    print("Enclave:")
    enclave.summary()

    new_filename = get_new_filename(model_file)

    print('\n')
    print('Saving model to {}'.format(new_filename))
    enclave_model.save(new_filename)

    print('Compiling enclave...')
    compile_enclave()

    print("Success!")

    return enclave_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build C files for sgx from TF model')
    parser.add_argument(
        'model_file', help='the .h5 file where the TF model is stored')
    parser.add_argument(
        'n', type=int, help='the number of layers to put in the sgx')
    # parser.add_argument(
    # 'output_dir', metavar='t', default='.', help='the output directory')

    args = parser.parse_args()
    model_file = args.model_file
    n = args.n

    breakpoint()
    build_enclave(model_file, n)
