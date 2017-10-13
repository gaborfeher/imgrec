#include "cnn/l2_error_layer.h"

L2ErrorLayer::L2ErrorLayer() {}

void L2ErrorLayer::SetExpectedValue(const DeviceMatrix& expected_value) {
  expected_value_ = expected_value;
}

void L2ErrorLayer::Forward(const DeviceMatrix& input) {
  expected_value_.AssertSameDimensions(input);
  input_ = input;
  output_ = input.Add(expected_value_.Multiply(-1)).L2();
}

void L2ErrorLayer::Backward(const DeviceMatrix& output_gradient) {
  // output_gradient is not used, we assume that this is the last
  // layer.
  output_gradient.AssertDimensions(0, 0, 0);
  float output = GetError();
  float multiplier = 0.0f;
  if (output != 0.0f) {
    multiplier = 1.0f / output;
  }

  input_gradients_ = input_.Add(expected_value_.Multiply(-1)).Multiply(multiplier);
}

float L2ErrorLayer::GetError() const {
  output_.AssertDimensions(1, 1, 1);
  return output_.GetVector()[0];
}

