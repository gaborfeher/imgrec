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

## Building and running

Make sure that the `NVCC` and `CUDA_LIB` variables in the `Makefile` are pointing to the correct locations.

1. Check this repo out.
2. Run `./download_deps.sh` to get GoogleTest and cereal.
3. Run `make` to run the tests, or run `make cifar10_train` to train a model on CIFAR-10 data.

# Sources

Some of the sources used:
* http://cs231n.stanford.edu/
* https://arxiv.org/abs/1502.03167
* http://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
* https://stats.stackexchange.com/questions/198840/cnn-xavier-weight-initialization
* https://developer.nvidia.com/cuda-toolkit
