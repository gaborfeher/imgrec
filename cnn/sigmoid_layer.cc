#include "cnn/sigmoid_layer.h"

SigmoidLayer::SigmoidLayer() {}

void SigmoidLayer::Forward(const DeviceMatrix& input) {
  input_ = input;
  output_ = input.ApplySigmoid();
}

void SigmoidLayer::Backward(const DeviceMatrix& output_gradients) {
  input_gradients_ = output_gradients.ApplySigmoidGradients();
}

void SigmoidLayer::ApplyGradient(float) {}
