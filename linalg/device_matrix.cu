#include "linalg/device_matrix.h"

#include <cassert>  // TODO: release-mode assert
#include <iostream>

#include <cuda_runtime.h>

DeviceMatrix::DeviceMatrix() : BaseMatrix(0, 0) {}

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

void DeviceMatrix::AssertSameDimensions(const DeviceMatrix& other) const {
  assert(rows_ == other.rows_ && cols_ == other.cols_);
}

void DeviceMatrix::AssertRows(int rows) const {
  assert(rows_ == rows);
}

__global__ void VecAdd(float* A, float* B, float* C) {
  int i = threadIdx.x;
  C[i] = A[i] + B[i];
}

DeviceMatrix DeviceMatrix::Add(const DeviceMatrix& other) const {
  AssertSameDimensions(other);
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

__global__ void VecMultiply(float* A, float m, float* B) {
  int i = threadIdx.x;
  B[i] = A[i] * m;
}

DeviceMatrix DeviceMatrix::Multiply(float m) const {
  DeviceMatrix result(rows_, cols_);
  VecMultiply<<<1, size_>>>(data_.get(), m, result.data_.get());
  return result;
}

__global__ void MatrixDotProd(
    float* A, int a_rows, int a_cols,
    float* B, int b_rows, int b_cols,
    float* C, int c_rows, int c_cols) {
  int i = threadIdx.x;
  int j = threadIdx.y;
  float sum = 0.0;
  for (int k = 0; k < a_cols; ++k) {
    sum += A[i * a_cols + k] * B[k * b_cols + j];
  }
  C[i * c_cols + j] = sum;
}

DeviceMatrix DeviceMatrix::Dot(const DeviceMatrix& other) const {
  assert(cols_ == other.rows_);
  int c_rows = rows_;
  int c_cols = other.cols_;
  DeviceMatrix result(c_rows, c_cols);
  dim3 grid(1, 1);
  dim3 threads(c_rows, c_cols);
  MatrixDotProd<<<grid, threads>>>(
      data_.get(), rows_, cols_,
      other.data_.get(), other.rows_, other.cols_,
      result.data_.get(), result.rows_, result.cols_);
  return result;
}

__global__ void VecSigmoid(float* A, float* B) {
  int i = threadIdx.x;
  B[i] = 1.0 / (1.0 + exp(-A[i]));
}

DeviceMatrix DeviceMatrix::ApplySigmoid() const {
  DeviceMatrix result(rows_, cols_);
  VecSigmoid<<<1, size_>>>(data_.get(), result.data_.get());
  return result;
}

__global__ void VecSigmoidGradients(float* A, float* B) {
  int i = threadIdx.x;
  float sigma = 1.0 / (1.0 + exp(-A[i]));
  B[i] = sigma * (1.0 - sigma);
}

DeviceMatrix DeviceMatrix::ApplySigmoidGradients() const {
  DeviceMatrix result(rows_, cols_);
  VecSigmoidGradients<<<1, size_>>>(data_.get(), result.data_.get());
  return result;
}

__global__ void VecL2(float* A, int len, float* B) {
  float result = 0.0;
  for (int i = 0; i < len; ++i) {
    result += A[i] * A[i];
  }
  B[0] = sqrt(result);
}

DeviceMatrix DeviceMatrix::L2() const {
  DeviceMatrix result(1, 1);
  VecL2<<<1, 1>>>(data_.get(), size_, result.data_.get());
  return result;
}
