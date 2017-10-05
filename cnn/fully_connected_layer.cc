#include "cnn/fully_connected_layer.h"

FullyConnectedLayer::FullyConnectedLayer(int input_size, int output_size) :
    input_size_(input_size),
    output_size_(output_size),
    weights_(output_size, input_size, 1),
    weights_gradients_(output_size, input_size, 1) {
  weights_.Fill(0.1f);  // TODO: fix, use Xavier-initialization
}

void FullyConnectedLayer::Forward(const DeviceMatrix& input) {
  input.AssertRows(input_size_);
  input_ = input;
  output_ = weights_.Dot(input);
}

void FullyConnectedLayer::Backward(const DeviceMatrix& output_gradients) {
  output_gradients.AssertRows(output_size_);
  weights_gradients_ = output_gradients.Dot(input_.T());
  input_gradients_ = weights_.T().Dot(output_gradients);
}

void FullyConnectedLayer::ApplyGradient(float learn_rate) {
  weights_ = weights_.Add(weights_gradients_.Multiply(-learn_rate));
}
