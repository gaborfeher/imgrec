#include "cnn/bias_layer.h"

#include <iostream>

#include "linalg/matrix.h"

BiasLayer::BiasLayer(int num_neurons, bool layered) :
    BiasLikeLayer(num_neurons, layered) {
  if (layered) {
    biases_= MatrixParam(1, 1, num_neurons);
  } else {
    biases_ = MatrixParam(num_neurons, 1, 1);
  }
}

void BiasLayer::Print() const {
  std::cout << "Bias Layer:" << std::endl;
  biases_.value.Print();
}

void BiasLayer::Initialize(Random*) {
  biases_.value.Fill(0);
}

void BiasLayer::Forward(const Matrix& input) {
  output_ = input.Add(biases_.value.Repeat(layered_, input));
}

void BiasLayer::Backward(const Matrix& output_gradient) {
  input_gradient_ = output_gradient;
  biases_.gradient =
      output_gradient.Sum(layered_, num_neurons_);
}

void BiasLayer::ApplyGradient(float learn_rate, float /* lambda */) {
  biases_.ApplyGradient(learn_rate, 0.0f);
}

int BiasLayer::NumParameters() const {
  return biases_.NumParameters();
}
