#include "cnn/fully_connected_layer.h"

FullyConnectedLayer::FullyConnectedLayer(int input_size, int output_size) :
    input_size_(input_size),
    output_size_(output_size),
    weights_(output_size, input_size),
    weights_gradients_(output_size, input_size) {
}

void FullyConnectedLayer::forward(const DeviceMatrix& input) {
  input.assert_rows(input_size_); 
  input_ = input;
  output_ = weights_.Dot(input);
}

void FullyConnectedLayer::backward(const DeviceMatrix& output_gradients) {
  output_gradients.assert_same_dimensions(output_);
  input_gradients_ = output_gradients.Dot(input_.T());
  weights_gradients_ = weights_.T.Dot(output_gradients);
}
