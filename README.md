# Features

A toy convolutional neural network training framework written from scratch using CUDA. Supported layers:
* Convolutional Layer
* Fully Connected Layer
* Constant Bias Layer (the above two are unbiased)
* Batch Normalization Layer
* Nonlinearity layer: Sigmoid, ReLU, LReLU
* (Inverted) Dropout Layer
* Max-pooling Layer (the pooling regions have to tile the image)
* Reshape Layer (to transition from Convoutional layers to Fully connected layers; most layers can handle both cases)
* L2 Error Layer
* Softmax Error Layer

All layers support handling batches of samples.

Optimization algorithms:
* SGD
* ADAM

Linear algebra: a `Matrix` class implements a lot of operations on top of CUDA. The operations are not too
heavily optimizied, but in return, easier to read.

# Results

The following file contains the model-building code for CIFAR-10:
https://github.com/gaborfeher/imgrec/blob/master/apps/cifar10/cifar10_train.cc

Detailed results are here:
https://github.com/gaborfeher/imgrec/tree/master/apps/cifar10/results

Accuracy of top scored result on the CIFAR-10 data set using the above models:

| Model | Validation | Test |
| --- | --- | --- |
| fc1 | 20.19% | TBD |
| fc2nodrop | 32.41% | TBD |
| fc2drop | 33.07% | TBD |
| conv1 | 59.33% | TBD |
| conv2 | 61.90% | TBD |

# Installation

## Prerequisites

* CUDA Toolkit (https://developer.nvidia.com/cuda-downloads)
* GoogleTest (https://github.com/google/googletest)
* cereal (https://github.com/USCiLab/cereal)
* Clang++

## Building and usage

1. Check out this repo.
2. Make sure to install GoogleTest's and cereal's prerequisites.
3. Run `./download_deps.sh` to get GoogleTest and cereal.
4. Check the values of `NVCC` and `CUDA_LIB` in the `Makefile` and fix them if needed.
5. Run `make` to run the tests.
6. Run `make cifar10_train` to train a model on CIFAR-10 data.
7. Run `make bin/apps/cifar10/use_model` if you want to try your
new model.
8. To evaluate the model on a data set, run something like
```
bin/apps/cifar10/score_image \
  ./apps/cifar10/results/conv2.2/epoch10.model \
  ./apps/cifar10/downloaded_deps/cifar-10-batches-bin/data_batch_5.bin
```
9. To feed an arbitrary image into the model, run something like:
```
apps/util/import_photo.py <your JPG file> img_xyz
bin/apps/cifar10/score_image \
  ./apps/cifar10/results/conv2.2/epoch10.model \
  ./img_xyz.bmp
```

# Sources

Some of the sources used:
* http://cs231n.stanford.edu/
* https://arxiv.org/abs/1502.03167
* http://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
* https://stats.stackexchange.com/questions/198840/cnn-xavier-weight-initialization
* https://developer.nvidia.com/cuda-toolkit
