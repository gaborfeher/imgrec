#!/bin/bash

make bin/apps/cifar10/cifar10_train
./bin/apps/cifar10/cifar10_train $1
