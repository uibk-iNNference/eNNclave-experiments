#!/bin/bash

source docker_setup.sh

python build_enclave.py models/mit.h5 3
python time_enclave.py models/mit_enclave.h5 3