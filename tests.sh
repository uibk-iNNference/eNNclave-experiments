#!/bin/bash

source /usr/local/etc/profile.d/conda.sh
conda activate eNNclave-experiments
source /opt/intel/sgxsdk/environment

pip install -e /eNNclave/frontend/python

python build_enclave models/amazon.h5 3