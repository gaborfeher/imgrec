#include "cnn/fully_connected_layer.h"

#include <random>
#include <vector>

// TODO: clean up dup
std::vector<float> GetRandomVector(int size, int seed) {
  std::mt19937 rnd(seed);
  std::uniform_real_distribution<> dist(-1, 1);

  // TODO: use Xavier-initialization
  std::vector<float> result;
  result.reserve(size);
  for (int i = 0; i < size; ++i) {
    result.push_back(dist(rnd));
  }
  return result;
}


FullyConnectedLayer::FullyConnectedLayer(int input_size, int output_size, int random_seed) :
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
      1,
      GetRandomVector(output_size * input_size_, random_seed));
  weights_gradients_ = DeviceMatrix(output_size, input_size_, 1);
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
