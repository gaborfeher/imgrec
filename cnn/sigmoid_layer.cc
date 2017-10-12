#include "cnn/sigmoid_layer.h"

SigmoidLayer::SigmoidLayer() {}

void SigmoidLayer::Forward(const DeviceMatrix& input) {
  input_ = input;
  output_ = input.Map(matrix_mappers::Sigmoid());
}

void SigmoidLayer::Backward(const DeviceMatrix& output_gradients) {
  input_gradients_ = input_.Map(matrix_mappers::SigmoidGradient()).ElementwiseMultiply(output_gradients);
}

void SigmoidLayer::ApplyGradient(float) {}
