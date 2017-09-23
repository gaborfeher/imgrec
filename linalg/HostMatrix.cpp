#include "HostMatrix.h"

#include <iostream>

#include <cuda_runtime.h>

#include "DeviceMatrix.h"

HostMatrix::HostMatrix(int rows, int cols, float* data) :
    BaseMatrix(rows, cols) {
  data_.reset(new float[size_], std::default_delete<float[]>() );
  for (int i = 0; i < size_; ++i) {
    data_.get()[i] = data[i];
  }
}

HostMatrix::HostMatrix(const DeviceMatrix& src) :
    BaseMatrix(src.rows_, src.cols_) {
  data_.reset(new float[size_], std::default_delete<float[]>() );
  cudaMemcpy(
      data_.get(),
      src.data_.get(),
      size_ * sizeof(float),
      cudaMemcpyDeviceToHost);
}

void HostMatrix::Print() const {
  std::cout << "size= " << size_ << std::endl;
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      std::cout << data_.get()[i * cols_ + j] << " ";
    }
    std::cout << std::endl;
  }
}

std::vector<float> HostMatrix::GetVector() const {
  std::vector<float> v;
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      v.push_back(data_.get()[i * cols_ + j]);
    }
  }
  return v;
}

