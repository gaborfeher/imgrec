#include "linalg/device_matrix.h"

#include <cassert>  // TODO: release-mode assert
#include <iostream>

#include <cuda_runtime.h>

#include "linalg/host_matrix.h"

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

DeviceMatrix DeviceMatrix::Add(const DeviceMatrix& other) const {
  assert(rows_ == other.rows_ && cols_ == other.cols_);
  DeviceMatrix result(rows_, cols_);
  VecAdd<<<1, size_>>>(data_.get(), other.data_.get(), result.data_.get());
  return result;
}

__global__ void MatrixTranspose(float* A, int rows_, int cols_, float* T) {
  int a_index = threadIdx.x * cols_ + threadIdx.y;
  int t_index = threadIdx.y * rows_ + threadIdx.x;
  T[t_index] = A[a_index];
}

DeviceMatrix DeviceMatrix::T() const {
  DeviceMatrix result(cols_, rows_);

  dim3 grid(1, 1); 
  dim3 threads(rows_, cols_);
  MatrixTranspose<<<grid, threads>>>(data_.get(), rows_, cols_, result.data_.get());
  return result;
}
