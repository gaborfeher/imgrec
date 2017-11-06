#include "cnn/l2_error_layer.h"

#include "cereal/archives/portable_binary.hpp"
#include "cereal/types/polymorphic.hpp"

L2ErrorLayer::L2ErrorLayer() {}

void L2ErrorLayer::SetExpectedValue(const Matrix& expected_value) {
  expected_value_ = expected_value;
}

void L2ErrorLayer::Forward(const Matrix& input) {
  expected_value_.AssertSameDimensions(input);
  input_ = input;
  error_ = input.Add(expected_value_.Multiply(-1)).L2();
}

void L2ErrorLayer::Backward(const Matrix& output_gradient) {
  // output_gradient is not used, we assume that this is the last
  // layer.
  output_gradient.AssertDimensions(0, 0, 0);

  input_gradient_ = input_
      .Add(expected_value_.Multiply(-1))
      .Multiply(2.0f);
}

float L2ErrorLayer::GetAccuracy() const {
  return -1.0;  // TODO
}

void L2ErrorLayer::save(cereal::PortableBinaryOutputArchive&) const {
}

void L2ErrorLayer::load(cereal::PortableBinaryInputArchive&) {
}

CEREAL_REGISTER_TYPE(L2ErrorLayer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(Layer, L2ErrorLayer);

