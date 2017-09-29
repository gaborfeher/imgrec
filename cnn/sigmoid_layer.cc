#include "cnn/sigmoid_layer.h"

SigmoidLayer::SigmoidLayer(int size) : size_(size) {}

void SigmoidLayer::forward(const DeviceMatrix& input) {
  input_ = input;
  output_ = input.ApplySigmoid();
}

void SigmoidLayer::backward(const DeviceMatrix& output_gradients) {
  input_gradients_ = output_gradients.ApplySigmoidGradients();
}
