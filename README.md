# Features

A toy convolutional neural network training framework written from scratch using CUDA. Supported layers:
* Convolutional Layer (padding is supported, striding always have to be one)
* Fully Connected Layer
* Nonlinearity layer: Sigmoid, ReLU, LReLU
* Constant Bias Layer (the above two are unbiased)
* Batch Normalization Layer
* (Inverted) Dropout Layer
* Pooling Layer (the pooling regions have to tile the image)
* Reshape Layer (to transition from Convoutional layers to Fully connected layers; most layers can handle both cases)
* L2 Error Layer
* Softmax Error Layer
All layers support handling batches of samples.

Optimization algorithms:
* SGD
* ADAM

Linear algebra: a `Matrix` class implements a lot of operations on top of CUDA. The operations are not too heavily optimizied,
but in return, easier to read.

See the following file for example model training scenarios for CIFAR-10: https://github.com/gaborfeher/imgrec/blob/master/apps/cifar10/cifar10_train.cc


# Installation

## Prerequisites

* https://developer.nvidia.com/cuda-downloads
* Clang++
* CMake and GNU C++ for building GoogleTest (downloading and building will happen automatically, but you need to install these)

## Building and running

Make sure that the `NVCC` and `CUDA_LIB` variables in the `Makefile` are pointing to the correct locations.

Check this repo out, say `make` to run the tests, or say `make cifar10_train` to train a model on CIFAR-10 data.

If the prerequisites were installed, then the first `make` run will download and build GoogleTest also.

# Sources

Some of the sources used:
* http://cs231n.stanford.edu/
* https://arxiv.org/abs/1502.03167
* http://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
* https://stats.stackexchange.com/questions/198840/cnn-xavier-weight-initialization
* https://developer.nvidia.com/cuda-toolkit
