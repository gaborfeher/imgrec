#include "cnn/l2_error_layer.h"

L2ErrorLayer::L2ErrorLayer() {}

void L2ErrorLayer::SetExpectedValue(const DeviceMatrix& expected_value) {
  expected_value_ = expected_value;
}

void L2ErrorLayer::Forward(const DeviceMatrix& input) {
  expected_value_.AssertSameDimensions(input);
  input_ = input;
  error_ = input.Add(expected_value_.Multiply(-1)).L2();
}

void L2ErrorLayer::Backward(const DeviceMatrix& output_gradient) {
  // output_gradient is not used, we assume that this is the last
  // layer.
  output_gradient.AssertDimensions(0, 0, 0);
  float multiplier = 0.0f;
  if (error_ != 0.0f) {
    multiplier = 1.0f / error_;
  }

  input_gradients_ = input_
      .Add(expected_value_.Multiply(-1))
      .Multiply(multiplier);
}

float L2ErrorLayer::GetAccuracy() const {
  return -1.0;  // TODO
}
