#include "linalg/device_matrix.h"

#include <cassert>  // TODO: release-mode assert
#include <iostream>

#include <cuda_runtime.h>

std::shared_ptr<float> AllocateData(int size) {
  float* data;
  cudaMalloc(&data, size * sizeof(float));
  return std::shared_ptr<float>(data, cudaFree);
}

DeviceMatrix::DeviceMatrix(int rows, int cols, float* data) :
    BaseMatrix(rows, cols) {
  data_ = AllocateData(size_);
  cudaMemcpy(
      data_.get(),
      data,
      size_ * sizeof(float),
      cudaMemcpyHostToDevice);
}


DeviceMatrix::DeviceMatrix(int rows, int cols) :
    BaseMatrix(rows, cols) {
  data_ = AllocateData(size_);
}

std::shared_ptr<float> DeviceMatrix::get_host_data() const {
  std::shared_ptr<float> host_data;
  host_data.reset(new float[size_], std::default_delete<float[]>() );
  cudaMemcpy(
      host_data.get(),
      data_.get(),
      size_ * sizeof(float),
      cudaMemcpyDeviceToHost);
  return host_data;
}

std::vector<float> DeviceMatrix::GetVector() const {
  std::shared_ptr<float> host_data(get_host_data());
  std::vector<float> v;
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      v.push_back(host_data.get()[i * cols_ + j]);
    }
  }
  return v;
}

void DeviceMatrix::Print() const {
  std::shared_ptr<float> host_data(get_host_data());
  std::cout << "size= " << size_ << std::endl;
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      std::cout << host_data.get()[i * cols_ + j] << " ";
    }
    std::cout << std::endl;
  }
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
