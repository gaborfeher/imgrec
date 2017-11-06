#include "cnn/reshape_layer.h"

#include <cassert>

#include "cereal/archives/portable_binary.hpp"
#include "cereal/types/memory.hpp"
#include "cereal/types/polymorphic.hpp"

#include "linalg/matrix.h"

ReshapeLayer::ReshapeLayer(int unit_rows, int unit_cols, int unit_depth) :
    unit_rows_(unit_rows),
    unit_cols_(unit_cols),
    unit_depth_(unit_depth) {}

void ReshapeLayer::Forward(const Matrix& input) {
  assert(input.rows() == unit_rows_);
  assert(input.cols() == unit_cols_);
  output_ = input.ReshapeToColumns(unit_depth_);
}

void ReshapeLayer::Backward(const Matrix& output_gradient) {
  input_gradient_ = output_gradient.ReshapeFromColumns(
      unit_rows_, unit_cols_, unit_depth_);
}

void ReshapeLayer::save(cereal::PortableBinaryOutputArchive& ar) const {
  ar(unit_rows_, unit_cols_, unit_depth_);
}

void ReshapeLayer::load(cereal::PortableBinaryInputArchive& ar) {
  ar(unit_rows_, unit_cols_, unit_depth_);
}

CEREAL_REGISTER_TYPE(ReshapeLayer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(Layer, ReshapeLayer);
