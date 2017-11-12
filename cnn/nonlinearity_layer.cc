#include "cnn/nonlinearity_layer.h"

#include <cassert>

#include "cereal/archives/portable_binary.hpp"
#include "cereal/types/memory.hpp"
#include "cereal/types/polymorphic.hpp"

namespace activation_functions {

ActivationFunc Sigmoid() {
  return std::make_pair(
      ::matrix_mappers::Sigmoid(),
      ::matrix_mappers::SigmoidGradient());
}

ActivationFunc ReLU() {
  return std::make_pair(
      ::matrix_mappers::ReLU(),
      ::matrix_mappers::ReLUGradient());
}

ActivationFunc LReLU() {
  return std::make_pair(
      ::matrix_mappers::LReLU(),
      ::matrix_mappers::LReLUGradient());
}

}  // activation_functions

NonlinearityLayer::NonlinearityLayer(
    ::activation_functions::ActivationFunc activation) :
  activation_function_(activation.first),
  activation_function_gradient_(activation.second) {}

std::string NonlinearityLayer::Name() const {
  return "NonlinearityLayer";
}

void NonlinearityLayer::Forward(const Matrix& input) {
  input_ = input;
  output_ = input.Map1(activation_function_);
}

void NonlinearityLayer::Backward(const Matrix& output_gradient) {
  input_gradient_ = input_
      .Map1(activation_function_gradient_)
      .ElementwiseMultiply(output_gradient);
}

void NonlinearityLayer::save(cereal::PortableBinaryOutputArchive& ar) const {
  std::string fun;
  if (activation_function_ == ::matrix_mappers::Sigmoid()) {
    fun = "Sigmoid";
  } else if (activation_function_ == ::matrix_mappers::ReLU()) {
    fun = "ReLU";
  } else if (activation_function_ == ::matrix_mappers::LReLU()) {
    fun = "LReLU";
  } else {
    assert(false);
  }
  ar(fun);
}

void NonlinearityLayer::load(cereal::PortableBinaryInputArchive& ar) {
  std::string fun;
  ar(fun);
  if (fun == "Sigmoid") {
    activation_function_ = ::matrix_mappers::Sigmoid();
    activation_function_gradient_ = ::matrix_mappers::Sigmoid();
  } else if (fun == "ReLU") {
    activation_function_ = ::matrix_mappers::ReLU();
    activation_function_gradient_ = ::matrix_mappers::ReLU();
  } else if (fun == "LReLU") {
    activation_function_ = ::matrix_mappers::LReLU();
    activation_function_gradient_ = ::matrix_mappers::LReLU();
  } else {
    assert(false);
  }
}

CEREAL_REGISTER_TYPE(NonlinearityLayer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(Layer, NonlinearityLayer);
