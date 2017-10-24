#include "cnn/bias_layer.h"

#include <iostream>

#include "linalg/device_matrix.h"

BiasLayer::BiasLayer(int num_neurons, bool layered) :
    BiasLikeLayer(num_neurons, layered) {
  if (layered) {
    biases_ = DeviceMatrix(1, 1, num_neurons);
    biases_gradient_ = DeviceMatrix(1, 1, num_neurons);
  } else {
    biases_ = DeviceMatrix(num_neurons, 1, 1);
    biases_gradient_ = DeviceMatrix(num_neurons, 1, 1);
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
      biases_.Repeat(layered_, input.rows(), input.cols(), input.depth()));
}

void BiasLayer::Backward(const DeviceMatrix& output_gradient) {
  input_gradient_ = output_gradient;
  biases_gradient_ = output_gradient.Sum(layered_, num_neurons_);
}

void BiasLayer::ApplyGradient(float learn_rate) {
  biases_ = biases_.Add(biases_gradient_.Multiply(-learn_rate));
}

