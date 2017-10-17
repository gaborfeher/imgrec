#include "cnn/fully_connected_layer.h"

#include <vector>
#include <random>

FullyConnectedLayer::FullyConnectedLayer(int input_size, int output_size) :
    bias_trick_(true),
    output_size_(output_size) {
  if (bias_trick_) {
    input_size_ = input_size + 1;
  } else {
    input_size_ = input_size;
  }
  weights_ = DeviceMatrix(
      output_size,
      input_size_,
      1);
  weights_gradients_ = DeviceMatrix(output_size, input_size_, 1);
}

void FullyConnectedLayer::Initialize(Random* random) {
  std::uniform_real_distribution<> dist(-1, 1);
  weights_.RandomFill(random, &dist);
}

void FullyConnectedLayer::Forward(const DeviceMatrix& input) {
  if (bias_trick_) {
    input_ = input.AddConstRow(1.0);  // simulate bias vector by this (aka. the bias trick)
  } else {
    input_ = input;
  }
  input_.AssertRows(input_size_);
  input_.AssertDepth(1);
  output_ = weights_.Dot(input_);
}

void FullyConnectedLayer::Backward(const DeviceMatrix& output_gradients) {
  output_gradients.AssertRows(output_size_);
  weights_gradients_ = output_gradients.Dot(input_.T());
  input_gradients_ = weights_
      .T()
      .Dot(output_gradients);
  if (bias_trick_) {
    input_gradients_ = input_gradients_
      .ReduceSize(input_.rows() - 1, input_.cols(), 1);
  }
}

void FullyConnectedLayer::ApplyGradient(float learn_rate) {
  weights_ = weights_.Add(weights_gradients_.Multiply(-learn_rate));
}

void FullyConnectedLayer::Regularize(float lambda) {
  // In case of bias_trick_, the biases are also regularized.
  weights_ = weights_.Multiply(1.0 - lambda);
}
