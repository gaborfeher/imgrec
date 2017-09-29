#include "cnn/error_layer.h"

ErrorLayer::ErrorLayer(const DeviceMatrix& expected_value) :
    Layer(expected_value.rows(), expected_value.cols(), 1, 1),
    expected_value_(expected_value) {}

void ErrorLayer::Forward(const DeviceMatrix& input) {
  input_ = input;
  output_ = input.Add(expected_value_.Multiply(-1)).L2();
}

void ErrorLayer::Backward(const DeviceMatrix&) {
  // TODO: use output_gradient?
  input_gradients_ = input_.Add(expected_value_.Multiply(-1)).Multiply(2);
}

void ErrorLayer::ApplyGradient(float) {
}
