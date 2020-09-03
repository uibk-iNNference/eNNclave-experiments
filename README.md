# Experiments for eNNclave paper

These are the experiments presented in our paper *eNNclave: offline inference with model confidentiality*.

## Setup

Initialize and update the eNNclave framework submodule with `git submodule update --init`.
The eNNclave framework uses an environment variable `ENNCLAVE_HOME` for finding the build structure.
This variable can be automatically set by sourcing the [*setup.sh*](setup.sh) script.

CUDA and python dependencies can be managed via *conda* using the provided [*environment.yml*](environment.yml).

The MNIST dataset is downloaded automatically using Keras.
The other datasets need to be downloaded manually: [flowers](https://www.kaggle.com/alxmamaev/flowers-recognition/data), [Amazon book reviews](http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Books.json.gz), [Amazon CD reviews](http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/CDs_and_Vinyl.json.gz).
