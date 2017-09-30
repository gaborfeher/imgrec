#include "cnn/fully_connected_layer.h"

FullyConnectedLayer::FullyConnectedLayer(int input_rows, int input_cols, int output_rows, int output_cols) :
    Layer(input_rows, input_cols, output_rows, output_cols),
    weights_(output_rows * output_cols, input_rows * input_cols),
    weights_gradients_(output_rows * output_cols, input_rows * input_cols) {
}

void FullyConnectedLayer::Forward(const DeviceMatrix& input) {
  input_ = input;
  output_ = weights_.Dot(input);
}

void FullyConnectedLayer::Backward(const DeviceMatrix& output_gradients) {
  input_gradients_ = output_gradients.Dot(input_.T());
  weights_gradients_ = weights_.T().Dot(output_gradients);
}

void FullyConnectedLayer::ApplyGradient(float learn_rate) {
  weights_ = weights_.Add(weights_gradients_.Multiply(-learn_rate));
}
