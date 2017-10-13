#include "cnn/softmax_error_layer.h"

SoftmaxErrorLayer::SoftmaxErrorLayer() {}

void SoftmaxErrorLayer::SetExpectedValue(const DeviceMatrix& expected_value) {
  expected_value_ = expected_value;
}

void SoftmaxErrorLayer::Forward(const DeviceMatrix& input) {
  input_ = input;
  error_ = input.Softmax(expected_value_);
}

void SoftmaxErrorLayer::Backward(const DeviceMatrix& output_gradient) {
  // output_gradient is not used, we assume that this is the last
  // layer.
  output_gradient.AssertDimensions(0, 0, 0);
  input_gradients_ = input_.SoftmaxGradient(expected_value_);
}

