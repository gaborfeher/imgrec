#include "cnn/bias_layer.h"

#include <iostream>

#include "linalg/device_matrix.h"

BiasLayer::BiasLayer(int neurons, bool convolutional) :
    convolutional_(convolutional) {
  if (convolutional) {
    biases_ = DeviceMatrix(1, 1, neurons);
    biases_gradient_ = DeviceMatrix(1, 1, neurons);
  } else {
    biases_ = DeviceMatrix(neurons, 1, 1);
    biases_gradient_ = DeviceMatrix(neurons, 1, 1);
  }
}

void BiasLayer::Print() const {
  std::cout << "Bias Layer:" << std::endl;
  biases_.Print();
}

void BiasLayer::Initialize(Random*) {
  biases_.Fill(0);
}

void BiasLayer::Forward(const DeviceMatrix& input) {
  input_ = input;
  output_ = input.Add(
      biases_.Repeat(input.rows(), input.cols(), input.depth()));
}

void BiasLayer::Backward(const DeviceMatrix& output_gradient) {
  input_gradients_ = output_gradient;
  if (convolutional_) {
    biases_gradient_ = output_gradient.Sum(biases_.depth());
  } else {
    biases_gradient_ = output_gradient.Sum(0);
  }
}

void BiasLayer::ApplyGradient(float learn_rate) {
  biases_ = biases_.Add(biases_gradient_.Multiply(-learn_rate));
}

