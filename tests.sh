#!/bin/bash

source /usr/local/etc/profile.d/conda.sh
conda activate eNNclave-experiments
source /opt/intel/sgxsdk/environment

(cd $ENNCLAVE_HOME/build && cmake ..)

pip install -e /eNNclave/frontend/python

python build_enclave.py models/amazon.h5 3
python time_enclave.py models/amazon_enclave.h5 3