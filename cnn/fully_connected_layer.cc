#include "cnn/fully_connected_layer.h"

#include <vector>
#include <iostream>
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

void FullyConnectedLayer::Print() const {
  std::cout << "Fully Connected Layer:" << std::endl;
  weights_.Print();
}

void FullyConnectedLayer::Initialize(Random* random) {
  // http://cs231n.github.io/neural-networks-2/#init
  int n = input_size_;
  if (bias_trick_) {
    n -= 1;
  }
  float variance = 2.0f / input_size_;
  std::normal_distribution<float> dist(0, sqrt(variance));
  weights_.RandomFill(random, &dist);
  if (bias_trick_) {
    weights_.FillColumn(input_size_ - 1, 0.0f);  // reset biases to zero
  }
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
  if (bias_trick_) {
    DeviceMatrix regularizer(weights_.rows(), weights_.cols(), 1);
    regularizer.Fill(1.0 - lambda);
    regularizer.FillColumn(regularizer.cols() - 1, 1.0f);
    weights_ = weights_.ElementwiseMultiply(regularizer);
  } else {
    weights_ = weights_.Multiply(1.0 - lambda);
  }
}
