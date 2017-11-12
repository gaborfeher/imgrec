#include "cnn/softmax_error_layer.h"

#include "cereal/archives/portable_binary.hpp"
#include "cereal/types/polymorphic.hpp"

SoftmaxErrorLayer::SoftmaxErrorLayer() {}

std::string SoftmaxErrorLayer::Name() const {
  return "SoftmaxErrorLayer";
}

void SoftmaxErrorLayer::SetExpectedValue(const Matrix& expected_value) {
  expected_value_ = expected_value;
}

void SoftmaxErrorLayer::Forward(const Matrix& input) {
  input_ = input;
  error_ = input.Softmax(expected_value_);
}

void SoftmaxErrorLayer::Backward(const Matrix& output_gradient) {
  // output_gradient is not used, we assume that this is the last
  // layer.
  output_gradient.AssertDimensions(0, 0, 0);
  input_gradient_ = input_.SoftmaxGradient(expected_value_);
}

float SoftmaxErrorLayer::GetAccuracy() const {
  return input_.NumMatches(expected_value_) / expected_value_.cols();
}

void SoftmaxErrorLayer::save(cereal::PortableBinaryOutputArchive&) const {
}

void SoftmaxErrorLayer::load(cereal::PortableBinaryInputArchive&) {
}

CEREAL_REGISTER_TYPE(SoftmaxErrorLayer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(Layer, SoftmaxErrorLayer);

