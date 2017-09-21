#include "DeviceMatrix.h"

#include <cassert>  // TODO: release-mode assert
#include <iostream>

#include <cuda_runtime.h>

#include "HostMatrix.h"

std::shared_ptr<float> AllocateData(int size) {
  float* data;
  cudaMalloc(&data, size * sizeof(float));
  return std::shared_ptr<float>(data, cudaFree);
}

DeviceMatrix::DeviceMatrix(const HostMatrix& src) :
    BaseMatrix(src.rows_, src.cols_) {
  data_ = AllocateData(size_);
  cudaMemcpy(
      data_.get(),
      src.data_.get(),
      size_ * sizeof(float),
      cudaMemcpyHostToDevice);
}

DeviceMatrix::DeviceMatrix(int rows, int cols) :
    BaseMatrix(rows, cols) {
  data_ = AllocateData(size_);
}

__global__ void VecAdd(float* A, float* B, float* C) {
  int i = threadIdx.x;
  C[i] = A[i] + B[i];
}

DeviceMatrix DeviceMatrix::Add(const DeviceMatrix& other) {
  assert(rows_ == other.rows_ && cols_ == other.cols_);
  DeviceMatrix result(rows_, cols_);
  VecAdd<<<1, size_>>>(data_.get(), other.data_.get(), result.data_.get());
  return result;
}

