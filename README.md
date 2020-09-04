# Experiments for eNNclave paper

These are the experiments presented in our paper *eNNclave: offline inference with model confidentiality*.

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