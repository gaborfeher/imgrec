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
    input_size_(input_size),
    output_size_(output_size),
    weights_(output_size, input_size, 1, GetRandomVector(output_size * input_size, random_seed)),
    weights_gradients_(output_size, input_size, 1) {
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
