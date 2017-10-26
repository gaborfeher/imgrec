#include "cnn/pooling_layer.h"

#include <iostream>

#include "linalg/matrix.h"

PoolingLayer::PoolingLayer(int pool_rows, int pool_cols) :
    pool_rows_(pool_rows),
    pool_cols_(pool_cols) {}

void PoolingLayer::Print() const {
  std::cout << "Pooling Layer" << std::endl;
}

void PoolingLayer::Forward(const Matrix& input) {
  std::pair<Matrix, Matrix> result = input.Pooling(
      pool_rows_, pool_cols_);
  output_ = result.first;
  switch_ = result.second;
}

void PoolingLayer::Backward(const Matrix& output_gradient) {
  input_gradient_ = output_gradient.PoolingSwitch(
      switch_, pool_rows_, pool_cols_);
}
