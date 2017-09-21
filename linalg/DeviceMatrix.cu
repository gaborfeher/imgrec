#include "DeviceMatrix.h"

#include "HostMatrix.h"

#include <cuda_runtime.h>

#include <iostream>

DeviceMatrix::DeviceMatrix(const HostMatrix& src) :
    rows_(src.rows_),
    cols_(src.cols_),
    size_(src.size_) {
  float* data;
  cudaMalloc(&data, size_ * sizeof(float));
  data_.reset(data, cudaFree);
  cudaMemcpy(
      data_.get(),
      src.data_.get(),
      size_ * sizeof(float),
      cudaMemcpyHostToDevice);
}

