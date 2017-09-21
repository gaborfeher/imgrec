#include "DeviceMatrix.h"

#include <iostream>

#include <cuda_runtime.h>

#include "HostMatrix.h"

DeviceMatrix::DeviceMatrix(const HostMatrix& src) :
    BaseMatrix(src.rows_, src.cols_) {
  float* data;
  cudaMalloc(&data, size_ * sizeof(float));
  data_.reset(data, cudaFree);
  cudaMemcpy(
      data_.get(),
      src.data_.get(),
      size_ * sizeof(float),
      cudaMemcpyHostToDevice);
}

