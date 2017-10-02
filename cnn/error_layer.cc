#include "cnn/error_layer.h"

ErrorLayer::ErrorLayer() {}

void ErrorLayer::SetExpectedValue(const DeviceMatrix& expected_value) {
  expected_value_ = expected_value;
}

void ErrorLayer::Forward(const DeviceMatrix& input) {
  expected_value_.AssertSameDimensions(input);
  input_ = input;
  output_ = input.Add(expected_value_.Multiply(-1)).L2();
}

void ErrorLayer::Backward(const DeviceMatrix&) {
  // TODO: use output_gradient?
  input_gradients_ = input_.Add(expected_value_.Multiply(-1)).Multiply(2);
}

void ErrorLayer::ApplyGradient(float) {
}
