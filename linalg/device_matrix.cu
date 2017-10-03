#include "linalg/device_matrix.h"

#include <cassert>  // TODO: release-mode assert
#include <iostream>

#include <cuda_runtime.h>

__device__ int Dim3toDim1(
    int i, int j, int k,
    int rows, int cols, int depth) {
  return k * rows * cols + i * cols + j;
}

int DeviceMatrix::Index(int i, int j, int k) const {
  return k * rows_ * cols_ + i * cols_ + j;
}

DeviceMatrix::DeviceMatrix() :
    rows_(0),
    cols_(0),
    depth_(0),
    size_(0) {}

std::shared_ptr<float> AllocateData(int size) {
  float* data;
  cudaMalloc(&data, size * sizeof(float));
  return std::shared_ptr<float>(data, cudaFree);
}

std::shared_ptr<float> ImportData(float size, float* host_data) {
  std::shared_ptr<float> device_data(AllocateData(size));
  cudaMemcpy(
      device_data.get(),
      host_data,
      size * sizeof(float),
      cudaMemcpyHostToDevice);
  return device_data;
}

DeviceMatrix::DeviceMatrix(int rows, int cols, float* data) :
    rows_(rows),
    cols_(cols),
    depth_(1),
    size_(rows * cols) {
  data_ = ImportData(size_, data);
}

DeviceMatrix::DeviceMatrix(int rows, int cols, int depth, float* data) :
    rows_(rows),
    cols_(cols),
    depth_(depth),
    size_(rows * cols * depth) {
  data_ = ImportData(size_, data);
}

DeviceMatrix::DeviceMatrix(int rows, int cols) :
    rows_(rows),
    cols_(cols),
    depth_(1),
    size_(rows * cols) {
  data_ = AllocateData(size_);
  Fill(0);
}

DeviceMatrix::DeviceMatrix(int rows, int cols, int depth) :
    rows_(rows),
    cols_(cols),
    depth_(depth),
    size_(rows * cols * depth) {
  data_ = AllocateData(size_);
  Fill(0);
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
  for (int k = 0; k < depth_; ++k) {
    for (int i = 0; i < rows_; ++i) {
      for (int j = 0; j < cols_; ++j) {
        v.push_back(host_data.get()[Index(i, j, k)]);
      }
    }
  }
  return v;
}

void DeviceMatrix::Print() const {
  std::shared_ptr<float> host_data(get_host_data());
  std::cout << "size= " << size_ << std::endl;
  for (int k = 0; k < depth_; ++k) {
    for (int i = 0; i < rows_; ++i) {
      for (int j = 0; j < cols_; ++j) {
        std::cout << host_data.get()[Index(i, j, k)] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}

void DeviceMatrix::AssertDimensions(int rows, int cols) const {
  assert(rows_ == rows && cols_ == cols);
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

__global__ void VecMult(float* A, float* B, float* C) {
  int i = threadIdx.x;
  C[i] = A[i] * B[i];
}

DeviceMatrix DeviceMatrix::ElementwiseMultiply(const DeviceMatrix& other) const {
  AssertSameDimensions(other);
  DeviceMatrix result(rows_, cols_);
  VecMult<<<1, size_>>>(data_.get(), other.data_.get(), result.data_.get());
  return result;
}

__global__ void MatrixTranspose(float* A, int rows_, int cols_, float* T) {
  int a_index = threadIdx.x * cols_ + threadIdx.y;
  int t_index = threadIdx.y * rows_ + threadIdx.x;
  T[t_index] = A[a_index];
}

DeviceMatrix DeviceMatrix::T() const {
  assert(depth_ == 1);
  DeviceMatrix result(cols_, rows_);

  dim3 grid(1, 1);
  dim3 threads(rows_, cols_);
  MatrixTranspose<<<grid, threads>>>(data_.get(), rows_, cols_, result.data_.get());
  return result;
}

__global__ void MatrixRot180(
    float* A,
    int rows_, int cols_, int depth_,
    float* R) {
  int a_index = Dim3toDim1(
      threadIdx.x, threadIdx.y, threadIdx.z,
      rows_, cols_, depth_);
  int r_index = Dim3toDim1(
      rows_ - threadIdx.x - 1, cols_ - threadIdx.y - 1, threadIdx.z,
      rows_, cols_, depth_);
  R[r_index] = A[a_index];
}

DeviceMatrix DeviceMatrix::Rot180() const {
  DeviceMatrix result(cols_, rows_, depth_);

  dim3 grid(1, 1, 1);
  dim3 threads(rows_, cols_, depth_);
  MatrixRot180<<<grid, threads>>>(
      data_.get(),
      rows_, cols_, depth_,
      result.data_.get());
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
  assert(cols_ == other.rows_  && depth_ == 1);
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

__global__ void VecFill(float value, float* A) {
  int i = threadIdx.x;
  A[i] = value;
}

void DeviceMatrix::Fill(float value) {
  VecFill<<<1, size_>>>(value, data_.get());
}

__global__ void MatrixPadding(
    float* A,
    int rows, int cols, int depth, int padding,
    float* B) {
  int i = threadIdx.x;
  int j = threadIdx.y;
  int k = threadIdx.z;

  int b_index = Dim3toDim1(
      i + padding, j + padding, k,
      rows + 2 * padding, cols + 2 * padding, depth);
  int a_index = Dim3toDim1(i, j, k, rows, cols, depth);
  B[b_index] = A[a_index];
}

DeviceMatrix DeviceMatrix::AddPadding(int padding) const {
  if (padding <= 0) {
    return *this;
  }
  
  DeviceMatrix result(
      rows_ + 2 * padding,
      cols_ + 2 * padding,
      depth_);  // filled with zeros

  dim3 grid(1, 1, 1);
  dim3 threads(rows_, cols_, depth_);
  MatrixPadding<<<grid, threads>>>(
      data_.get(), rows_, cols_, depth_, padding,
      result.data_.get());
  return result;
}

__global__ void MatrixConvolution(
    int layers_per_image,
    float* A, int a_rows, int a_cols, int a_depth,
    float* filters, int f_rows, int f_cols, int f_depth,
    float* B, int b_rows, int b_cols, int b_depth) {
  int i = threadIdx.x;
  int j = threadIdx.y;
  int k = threadIdx.z;  // destination depth-level = id of filter to apply

  // layout of resulting matrix (list of layers):
  //
  // 1st image with 1st filter
  // 1st image with 2nd filter
  // ...
  // 2nd image with 1st filter
  // 2nd image with 2nd filter
  // ...

  int num_filters = f_depth / layers_per_image;
  int filter_id = k % num_filters;
  int image_id = k / num_filters;


  float sum = 0.0;
  for (int fk = 0; fk < layers_per_image; ++fk) {
    for (int fi = 0; fi < f_rows; ++fi) {
      for (int fj = 0; fj < f_cols; ++fj) {
        int filter_index = Dim3toDim1(
            fi,
            fj,
            fk + filter_id * layers_per_image,  // fk: level in cur. filter
            f_rows, f_cols, f_depth);
        int a_index = Dim3toDim1(
            i + fi,
            j + fj,
            fk + image_id * layers_per_image,  // fk: level in cur. image
            a_rows, a_cols, a_depth);

        sum += filters[filter_index] * A[a_index];
      }
    }
  }
  B[Dim3toDim1(i, j, k, b_rows, b_cols, b_depth)] = sum;
}

DeviceMatrix DeviceMatrix::Convolution(
    const DeviceMatrix& filters,
    int layers_per_image,
    int stride) const {
  int row_slots = rows_ - filters.rows() + 1;
  int col_slots = cols_ - filters.cols() + 1;
  assert(row_slots % stride == 0 && col_slots % stride == 0);

  assert(filters.depth() % layers_per_image == 0);
  assert(depth() % layers_per_image == 0);

  assert(stride == 1);  // TODO
  DeviceMatrix result(
      row_slots / stride,
      col_slots / stride,
      filters.depth() / layers_per_image * depth() / layers_per_image);
  dim3 grid(1, 1, 1);
  dim3 threads(result.rows(), result.cols(), result.depth());
  MatrixConvolution<<<grid, threads>>>(
      layers_per_image,
      data_.get(), rows_, cols_, depth_,
      filters.data_.get(), filters.rows(), filters.cols(), filters.depth(),
      result.data_.get(), result.rows(), result.cols(), result.depth());

  return result;
}
