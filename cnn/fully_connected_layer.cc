#include "cnn/fully_connected_layer.h"

#include <vector>
#include <iostream>
#include <random>

#include "linalg/matrix.h"

FullyConnectedLayer::FullyConnectedLayer(int input_size, int output_size) :
    input_size_(input_size),
    output_size_(output_size),
    weights_(output_size, input_size_, 1) {}

void FullyConnectedLayer::Print() const {
  std::cout << "Fully Connected Layer:" << std::endl;
  weights_.value.Print();
}

void FullyConnectedLayer::Initialize(Random* random) {
  // http://cs231n.github.io/neural-networks-2/#init
  float variance = 2.0f / input_size_;
  std::normal_distribution<float> dist(0, sqrt(variance));
  weights_.value.RandomFill(random, &dist);
}

void FullyConnectedLayer::Forward(const Matrix& input) {
  input_ = input;
  input_.AssertRows(input_size_);
  input_.AssertDepth(1);
  output_ = weights_.value.Dot(input_);
}

void FullyConnectedLayer::Backward(const Matrix& output_gradient) {
  output_gradient.AssertRows(output_size_);
  weights_.gradient = output_gradient.Dot(input_.T());

  input_gradient_ = weights_.value
      .T()
      .Dot(output_gradient);
}

void FullyConnectedLayer::ApplyGradient(const GradientInfo& info) {
  weights_.ApplyGradient(info);
}

int FullyConnectedLayer::NumParameters() const {
  return weights_.NumParameters();
}

