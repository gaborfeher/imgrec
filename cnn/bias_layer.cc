#include "cnn/bias_layer.h"

#include <iostream>

#include "cereal/archives/portable_binary.hpp"
#include "cereal/types/memory.hpp"
#include "cereal/types/polymorphic.hpp"

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

void BiasLayer::ApplyGradient(const GradientInfo& info) {
  GradientInfo copy = info;
  copy.lambda = 0.0f;  // no regularization
  biases_.ApplyGradient(copy);
}

int BiasLayer::NumParameters() const {
  return biases_.NumParameters();
}

void BiasLayer::save(cereal::PortableBinaryOutputArchive& ar) const {
  ar(num_neurons_, layered_, biases_);
}

void BiasLayer::load(cereal::PortableBinaryInputArchive& ar) {
  ar(num_neurons_, layered_, biases_);
}

CEREAL_REGISTER_TYPE(BiasLayer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(Layer, BiasLayer);

