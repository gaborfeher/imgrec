#include "cnn/nonlinearity_layer.h"

namespace activation_functions {

ActivationFunc Sigmoid() {
  return std::make_pair(
      ::matrix_mappers::Sigmoid(),
      ::matrix_mappers::SigmoidGradient());
}

ActivationFunc ReLU() {
  return std::make_pair(
      ::matrix_mappers::ReLU(),
      ::matrix_mappers::ReLUGradient());
}

ActivationFunc LReLU() {
  return std::make_pair(
      ::matrix_mappers::LReLU(),
      ::matrix_mappers::LReLUGradient());
}

}  // activation_functions

NonlinearityLayer::NonlinearityLayer(
    ::activation_functions::ActivationFunc activation) :
  activation_function_(activation.first),
  activation_function_gradient_(activation.second) {}

void NonlinearityLayer::Forward(const DeviceMatrix& input) {
  input_ = input;
  output_ = input.Map(activation_function_);
}

void NonlinearityLayer::Backward(const DeviceMatrix& output_gradients) {
  input_gradients_ = input_
      .Map(activation_function_gradient_)
      .ElementwiseMultiply(output_gradients);
}

void NonlinearityLayer::ApplyGradient(float) {}
