#include "cnn/reshape_layer.h"
#include "linalg/device_matrix.h"

ReshapeLayer::ReshapeLayer(int unit_rows, int unit_cols, int unit_depth) :
    unit_rows_(unit_rows),
    unit_cols_(unit_cols),
    unit_depth_(unit_depth) {}

void ReshapeLayer::Forward(const DeviceMatrix& input) {
  input.Print();
  output_ = input.ReshapeToColumns(unit_depth_);
}

void ReshapeLayer::Backward(const DeviceMatrix& output_gradients) {
  input_gradients_ = output_gradients.ReshapeFromColumns(
      unit_rows_, unit_cols_, unit_depth_);
}

void ReshapeLayer::ApplyGradient(float) {
}
