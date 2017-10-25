#include "cnn/reshape_layer.h"
#include "linalg/matrix.h"

ReshapeLayer::ReshapeLayer(int unit_rows, int unit_cols, int unit_depth) :
    unit_rows_(unit_rows),
    unit_cols_(unit_cols),
    unit_depth_(unit_depth) {}

void ReshapeLayer::Forward(const Matrix& input) {
  output_ = input.ReshapeToColumns(unit_depth_);
}

void ReshapeLayer::Backward(const Matrix& output_gradient) {
  input_gradient_ = output_gradient.ReshapeFromColumns(
      unit_rows_, unit_cols_, unit_depth_);
}

void ReshapeLayer::ApplyGradient(float) {
}
