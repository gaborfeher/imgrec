#include "cnn/error_layer.h"

ErrorLayer::ErrorLayer(const DeviceMactirx& expected_value) :
    expected_value_(expected_value) {}

ErrorLayer::Forward(const DeviceMatrix& input) {
  input_ = inpput;
  output_ = input.Add(expected_value_.Multiply(-1)).L2();
}

ErrorLayer::Backward(const DeviceMatrix& output_gradients) {
  input_gradients_ = input_.Add(expected_value_.Multiply(-1)).Multiply(2);
}

ErrorLayer::ApplyGradient(float learn_rate) {
}
