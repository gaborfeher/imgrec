#include <iostream>

#include "HostMatrix.h"

#include <cuda_runtime.h>

#include "DeviceMatrix.h"

HostMatrix::HostMatrix(int rows, int cols, float* data) :
    rows_(rows),
    cols_(cols),
    size_(rows * cols),
    data_(new float[size_], std::default_delete<float[]>() ) {
  for (int i = 0; i < size_; ++i) {
    data_.get()[i] = data[i];
  }
}

HostMatrix::HostMatrix(const DeviceMatrix& src) :
    rows_(src.rows_),
    cols_(src.cols_),
    size_(src.size_),
    data_(new float[size_], std::default_delete<float[]>() ) {
  cudaMemcpy(
      data_.get(),
      src.data_.get(),
      size_ * sizeof(float),
      cudaMemcpyDeviceToHost);
}

void HostMatrix::Print() {
  std::cout << "size= " << size_ << std::endl;
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      std::cout << data_.get()[i * cols_ + j] << " ";
    }
    std::cout << std::endl;
  }
}



