#include "cnn/fully_connected_layer.h"

#include <vector>
#include <iostream>
#include <random>

FullyConnectedLayer::FullyConnectedLayer(int input_size, int output_size) :
    input_size_(input_size),
    output_size_(output_size),
    weights_(
        output_size,
        input_size_,
        1),
    weights_gradient_(output_size, input_size_, 1) {}

void FullyConnectedLayer::Print() const {
  std::cout << "Fully Connected Layer:" << std::endl;
  weights_.Print();
}

void FullyConnectedLayer::Initialize(Random* random) {
  // http://cs231n.github.io/neural-networks-2/#init
  float variance = 2.0f / input_size_;
  std::normal_distribution<float> dist(0, sqrt(variance));
  weights_.RandomFill(random, &dist);
}

void FullyConnectedLayer::Forward(const DeviceMatrix& input) {
  input_ = input;
  input_.AssertRows(input_size_);
  input_.AssertDepth(1);
  output_ = weights_.Dot(input_);
}

void FullyConnectedLayer::Backward(const DeviceMatrix& output_gradient) {
  output_gradient.AssertRows(output_size_);
  weights_gradient_ = output_gradient.Dot(input_.T());

  input_gradient_ = weights_
      .T()
      .Dot(output_gradient);
}

void FullyConnectedLayer::ApplyGradient(float learn_rate) {
  weights_ = weights_.Add(weights_gradient_.Multiply(-learn_rate));
}

void FullyConnectedLayer::Regularize(float lambda) {
  weights_ = weights_.Multiply(1.0 - lambda);
}
