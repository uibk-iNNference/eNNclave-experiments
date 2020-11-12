# Experiments for eNNclave paper

These are the experiments presented in our paper [eNNclave: offline inference with model confidentiality](https://dl.acm.org/doi/10.1145/3411508.3421376).

## Setup

Initialize and update the eNNclave framework submodule with `git submodule update --init`.
The eNNclave framework uses an environment variable `ENNCLAVE_HOME` for finding the build structure.
This variable can be automatically set by sourcing the [*setup.sh*](setup.sh) script.

CUDA and python dependencies can be managed via *conda* using the provided [*environment.yml*](environment.yml).

The MNIST dataset is downloaded automatically using Keras.
The other datasets need to be downloaded manually: [flowers](https://www.kaggle.com/alxmamaev/flowers-recognition/data), [Amazon book reviews](http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Books.json.gz), [Amazon CD reviews](http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/CDs_and_Vinyl.json.gz).

### Obtaining the pre-trained VGG-16 model on the Places-365 standard model

The [Places-365 repository](https://github.com/CSAILVision/places365) contains weights for a VGG-16 model.
Unfortunately, these weights are only available in the Caffe format.
To convert this into a (for us) usable format we used Microsoft's [MMdnn](https://github.com/Microsoft/MMdnn), which is conveniently provided as a Docker image.
The Docker image can be pulled with `pip install -U git+https://github.com/Microsoft/MMdnn.git@master`.

#### Converting the model using MMdnn
First download the [caffe deploy file](https://github.com/CSAILVision/places365/blob/master/deploy_vgg16_places365.prototxt) and the [caffe weights](http://places2.csail.mit.edu/models_places365/vgg16_places365.caffemodel).
For the rest of this section I will assume you stored them in `/tmp`, but you can place them wherever you desire (just be sure to mount the correct directory to your docker container).

Then, start your docker container and mount the directory you stored the downloaded files into using the command `docker run -v /tmp:/tmp -it mmdnn/mmdnn:cpu.small`.
The `-v` flag and its arguments mount the local `/tmp` directory into the docker container, while the `-it` ensures we have a terminal to interact with.

Inside the docker container you can convert the model using the `mmconvert` command.
The full command line is as follows:
```shell script
mmconvert --srcFramework caffe --inputNetwork deploy_vgg16_places365.prototxt --inputWeight vgg16_places365.caffemodel --dstFramework keras --outputModel vgg16_places365.h5
```
After the command is done, there should be a `vgg16_places365.h5` file in your `/tmp` directory, which you can copy to the models directory and continue from there.

### Obtaining the datasets

The datasets can be obtained from the following links:
 - [MNIST](http://yann.lecun.com/exdb/mnist/)
 - [MIT67](http://web.mit.edu/torralba/www/indoor.html)
 - [Amazon Review Data](https://jmcauley.ucsd.edu/data/amazon/)
 - [Flowers](https://www.kaggle.com/alxmamaev/flowers-recognition/data)

### Setting up the eNNclave framework

The instructions for setting up the framework can be found in the [eNNclave repository](https://github.com/alxshine/eNNclave)

## Training the models

Here are the instructions for training the models.
Seeds should be set in all training scripts and the location for downloading our models can be found at the end of this section.
Due to the compartmentalization of the scripts you need to call them as models, and not as scripts directly.
For example, running the [mnist/train.py](mnist/train.py) script, you need to call it as
```shell
python -m mnist.train
```

### MNIST

The MNIST model is not used in the paper, but it is a good model to test the toolchain and environment.
Training the model is done via the [mnist/train](mnist/train.py) script.

### Flowers

The flower model can be trained using the [flowers/train](flowers/train.py) script.

### MIT-67

Training both accuracy variants for the MIT is done automatically in the [mit/train](mit/train.py) script.

### Amazon

Again, use the [amazon/train](amazon/train.py) script.

### Obtaining our models

Our parameter sets for the model weights can be found [here](https://ifi-nabu.uibk.ac.at/index.php/s/AFdf6CmxHntAQeb)

## Accuracy evaluation

Once you have the trained models, you can use the [mit/evaluate_accuracy](mit/evaluate_accuracy.py) script to evaluate the accuracy of the frozen and unfrozen variants of the model.

## Performance evaluation

First ensure that you have correctly set up the [eNNclave framework](https://github.com/alxshine/eNNclave).
Also check that `ENNCLAVE_HOME` and the `LD_LIBRARY_PATH` are correctly set.
Then you can use the [build_enclave](build_enclave.py) and [time_enclave](time_enclave.py) scripts for building the enclave and timing the execution.

Our [batch_time.sh](batch_time.sh) script automates this process for a given model and a number of splits.
Note that when passing the desired model to the shell script, please leave out the `.h5` file ending.

## Running the experiments in Docker

There is a Dockerfile for running the experiments, but that's not 100% stable right now.
I will include a shell script for launching Docker with mounts for the model and datasets directory ASAP (I promise).