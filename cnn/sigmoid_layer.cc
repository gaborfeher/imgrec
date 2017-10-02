#include "cnn/sigmoid_layer.h"

SigmoidLayer::SigmoidLayer() {}

void SigmoidLayer::Forward(const DeviceMatrix& input) {
  input_ = input;
  output_ = input.ApplySigmoid();
}

void SigmoidLayer::Backward(const DeviceMatrix& output_gradients) {
  input_gradients_ = input_.ApplySigmoidGradients().ElementwiseMultiply(output_gradients);
}

void SigmoidLayer::ApplyGradient(float) {}
