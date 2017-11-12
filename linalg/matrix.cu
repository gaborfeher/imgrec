#include "linalg/matrix.h"

#include "linalg/cuda_util.h"

#include <cassert>  // TODO: release-mode assert
#include <iostream>
#include <iomanip>
#include <math.h>

#include <cuda.h>  // strangely, not needed by nvcc
#include <curand.h>

#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>

namespace matrix_cuda_util {

void SynchronizeForPerfLogging() {
  cudaDeviceSynchronize();
}

}

int Matrix::Index(int i, int j, int k) const {
  return k * rows_ * cols_ + i * cols_ + j;
}

struct MatrixPack {
  float* items;
  int rows;
  int cols;
  int depth;
  int layer_size;

  explicit MatrixPack(const Matrix& m) :
      items(m.data_.get()),
      rows(m.rows()),
      cols(m.cols()),
      depth(m.depth()),
      layer_size(m.rows() * m.cols()) {}

  __forceinline__ __device__ float get(int i, int j, int k) {
    return items[k * layer_size + i * cols + j];
  }

  __forceinline__ __device__ float get(int i, int j) {
    return items[i * cols + j];
  }

  __forceinline__ __device__ void set(int i, int j, int k, float f) {
    items[k * layer_size + i * cols + j] = f;
  }

  __forceinline__ __device__ void set(int i, int j, float f) {
    items[i * cols + j] = f;
  }

  __forceinline__ __device__ void div(int i, int j, float f) {
    items[i * cols + j] /= f;
  }

  __forceinline__ __device__ void add(int i, int j, float a) {
    items[i * cols + j] += a;
  }

  __forceinline__ __device__ bool inside(int i, int j, int k) {
    return i < rows && j < cols && k < depth;
  }

  __forceinline__ __device__ bool inside(int i, int j) {
    return i < rows && j < cols;
  }

};

dim3 CalculateBlocks(
    const Matrix& result,
    dim3 threads_per_block) {
  return dim3(
      (result.rows() + threads_per_block.x - 1) / threads_per_block.x,
      (result.cols() + threads_per_block.y - 1) / threads_per_block.y,
      (result.depth() + threads_per_block.z - 1) / threads_per_block.z);
}

Matrix::Matrix() :
    rows_(0),
    cols_(0),
    depth_(0),
    size_(0),
    data_(NULL) {}

std::shared_ptr<float> AllocateData(int size) {
  float* data;
  CUDA_CALL(cudaMalloc(&data, size * sizeof(float)));
  return std::shared_ptr<float>(data, cudaFree);
}

std::shared_ptr<float> ImportData(int size, const float* host_data) {
  std::shared_ptr<float> device_data(AllocateData(size));
  CUDA_CALL(cudaMemcpy(
      device_data.get(),
      host_data,
      size * sizeof(float),
      cudaMemcpyHostToDevice));
  CUDA_ASYNC_CHECK();
  return device_data;
}

Matrix::Matrix(int rows, int cols, int depth, const std::vector<float>& data) :
    rows_(rows),
    cols_(cols),
    depth_(depth),
    size_(rows * cols * depth) {
  SetVector(data);
}

Matrix::Matrix(int rows, int cols, int depth) :
    rows_(rows),
    cols_(cols),
    depth_(depth),
    size_(rows * cols * depth) {
  data_ = AllocateData(size_);
}

std::shared_ptr<float> Matrix::get_host_data() const {
  std::shared_ptr<float> host_data;
  host_data.reset(new float[size_], std::default_delete<float[]>() );
  CUDA_CALL(cudaMemcpy(
      host_data.get(),
      data_.get(),
      size_ * sizeof(float),
      cudaMemcpyDeviceToHost));
  return host_data;
}

void Matrix::SetVector(const std::vector<float>& data) {
  assert(data.size() == size_);
  data_ = ImportData(size_, &data[0]);
}

std::vector<float> Matrix::GetVector() const {
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

void Matrix::Print() const {
  std::cout << std::fixed << std::setw(4) << std::setprecision(2);
  std::shared_ptr<float> host_data(get_host_data());
  std::cout << "Matrix "
      << rows_ << "x"
      << cols_ << "x"
      << depth_
      << " (" << size_ << ")" << std::endl;
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

void Matrix::AssertEquals(const Matrix& other) const {
  AssertSameDimensions(other);
  assert(GetVector() == other.GetVector());
}

void Matrix::AssertDimensions(int rows, int cols, int depth) const {
  assert(rows_ == rows && cols_ == cols && depth_ == depth);
}

void Matrix::AssertSameDimensions(const Matrix& other) const {
  assert(rows_ == other.rows_ && cols_ == other.cols_ && depth_ == other.depth_);
}

void Matrix::AssertRows(int rows) const {
  assert(rows_ == rows);
}

void Matrix::AssertDepth(int depth) const {
  assert(depth_ == depth);
}

__global__ void MatrixTranspose(MatrixPack a, MatrixPack t) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  if (a.inside(i, j)) {
    t.set(j, i, a.get(i, j));
  }
}

Matrix Matrix::T() const {
  assert(depth_ == 1);
  Matrix result(cols_, rows_, depth_);

  dim3 threads_per_block(16, 16, 1);
  dim3 blocks = CalculateBlocks(*this, threads_per_block);
  MatrixTranspose<<<blocks, threads_per_block>>>(
      MatrixPack(*this), MatrixPack(result));
  CUDA_ASYNC_CHECK();
  return result;
}

__global__ void MatrixRot180(
    MatrixPack a, MatrixPack r) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int k = threadIdx.z + blockDim.z * blockIdx.z;
  if (a.inside(i, j, k)) {
    r.set(r.rows - i - 1, r.cols - j - 1, k, a.get(i, j, k));
  }
}

Matrix Matrix::Rot180() const {
  Matrix result(rows_, cols_, depth_);
  dim3 threads_per_block(16, 16, 1);
  dim3 blocks = CalculateBlocks(result, threads_per_block);
  MatrixRot180<<<blocks, threads_per_block>>>(
      MatrixPack(*this),
      MatrixPack(result));
  return result;
}

__global__ void MatrixDotProd(
    MatrixPack a,
    MatrixPack b,
    MatrixPack c) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  if (c.inside(i, j)) {
    float sum = 0.0;
    for (int k = 0; k < a.cols; ++k) {
      sum += a.get(i, k) * b.get(k, j);
    }
    c.set(i, j, sum);
  }
}

Matrix Matrix::Dot(const Matrix& other) const {
  assert(cols_ == other.rows_);
  assert(depth_ == 1);
  int c_rows = rows_;
  int c_cols = other.cols_;
  Matrix result(c_rows, c_cols, 1);
  dim3 threads_per_block(16, 16, 1);
  dim3 blocks = CalculateBlocks(result, threads_per_block);
  MatrixDotProd<<<blocks, threads_per_block>>>(
      MatrixPack(*this),
      MatrixPack(other),
      MatrixPack(result));
  CUDA_ASYNC_CHECK();
  return result;
}

__global__ void VecSigmoid(float* a, float* b, int size) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    b[i] = 1.0 / (1.0 + exp(-a[i]));
  }
}

__global__ void VecSigmoidGradient(float* a, float*b, int size) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    float sigma = 1.0 / (1.0 + exp(-a[i]));
    b[i] = sigma * (1.0 - sigma);
  }
}

__global__ void VecReLU(float* a, float* b, int size) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    b[i] = max(0.0f, a[i]);
  }
}

__global__ void VecReLUGradient(float* a, float* b, int size) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    if (a[i] < 0.0f) {
      b[i] = 0.0f;
    } else {
      b[i] = 1.0f;
    }
  }
}

__global__ void VecLReLU(float* a, float* b, int size) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    b[i] = max(0.01f * a[i], a[i]);
  }
}

__global__ void VecLReLUGradient(float* a, float* b, int size) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    if (a[i] < 0.0f) {
      b[i] = 0.01f;
    } else {
      b[i] = 1.0f;
    }
  }
}

__global__ void VecSquare(float* a, float* b, int size) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    b[i] = a[i] * a[i];
  }
}

__global__ void VecSqrt(float* a, float* b, int size) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    b[i] = sqrt(a[i]);
  }
}

namespace matrix_mappers {

// We provide factory methdos instead of direct implementations
// so that users of device_matrix.h won't need to depend on
// CUDA stuff.

Map1Func Sigmoid() {
  return &VecSigmoid;
}

Map1Func SigmoidGradient() {
  return &VecSigmoidGradient;
}

Map1Func ReLU() {
  return &VecReLU;
}

Map1Func ReLUGradient() {
  return &VecReLUGradient;
}

Map1Func LReLU() {
  return &VecLReLU;
}

Map1Func LReLUGradient() {
  return &VecLReLUGradient;
}

Map1Func Square() {
  return &VecSquare;
}

Map1Func Sqrt() {
  return &VecSqrt;
}

}  // namespacce matrix_mappers

Matrix Matrix::Map1(::matrix_mappers::Map1Func map) const {
  Matrix result(rows_, cols_, depth_);
  map<<<(size_ + 255) / 256, 256>>>(
      data_.get(),
      result.data_.get(),
      size_);
  CUDA_ASYNC_CHECK();
  return result;
}

__global__ void ElementwiseAddKernel(float* a, float* b, float* c, int size) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    c[i] = a[i] + b[i];
  }
}

__global__ void ElementwiseMultiplyKernel(float* a, float* b, float* c, int size) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    c[i] = a[i] * b[i];
  }
}

__global__ void ElementwiseDivideKernel(float* a, float* b, float* c, int size) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    c[i] = a[i] / b[i];
  }
}

Matrix Matrix::Map2(const Matrix& other, ::matrix_mappers::Map2Func map) const {
  AssertSameDimensions(other);
  Matrix result(rows_, cols_, depth_);
  map<<<(size_ + 255) / 256, 256>>>(
      data_.get(), other.data_.get(), result.data_.get(), size_);
  CUDA_ASYNC_CHECK();
  return result;
}

Matrix Matrix::Add(const Matrix& other) const {
  return Map2(other, ElementwiseAddKernel);
}

Matrix Matrix::ElementwiseMultiply(const Matrix& other) const {
  return Map2(other, ElementwiseMultiplyKernel);
}

Matrix Matrix::ElementwiseDivide(const Matrix& other) const {
  return Map2(other, ElementwiseDivideKernel);
}

__global__ void AddConstKernel(float* a, float* b, int size, float c) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    b[i] = a[i] + c;
  }
}

__global__ void PowConstKernel(float* a, float* b, int size, float exp) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    b[i] = pow(a[i], exp);
  }
}

__global__ void MultiplyConstKernel(float* a, float* b, int size, float m) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    b[i] = a[i] * m;
  }
}

__global__ void DivideConstKernel(float* a, float* b, int size, float d) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    b[i] = a[i] / d;
  }
}

Matrix Matrix::Map1P(float param, ::matrix_mappers::Map1PFunc map) const {
  Matrix result(rows_, cols_, depth_);
  map<<<(size_ + 255) / 256, 256>>>(
      data_.get(), result.data_.get(), size_, param);
  CUDA_ASYNC_CHECK();
  return result;
}

Matrix Matrix::AddConst(float c) const {
  return Map1P(c, AddConstKernel);
}

Matrix Matrix::Pow(float exp) const {
  return Map1P(exp, PowConstKernel);
}

Matrix Matrix::Multiply(float m) const {
  return Map1P(m, MultiplyConstKernel);
}

Matrix Matrix::Divide(float m) const {
  return Map1P(m, DivideConstKernel);
}

__global__ void MatrixSumLayers(
    MatrixPack a, MatrixPack b) {
  int b_index = threadIdx.x + blockDim.x * blockIdx.x;
  if (b_index < b.depth) {
    float result = 0.0;
    for (int k = b_index; k < a.depth; k += b.depth) {
      for (int i = 0; i < a.rows; ++i) {
        for (int j = 0; j < a.cols; ++j) {
          result += a.get(i, j, k);
        }
      }
    }
    b.items[b_index] = result;
  }
}

float Matrix::Sum() const {
  Matrix result(1, 1, 1);
  MatrixSumLayers<<<1, 1>>>(MatrixPack(*this), MatrixPack(result));
  CUDA_ASYNC_CHECK();
  return result.GetValue(0, 0, 0);
}

__global__ void MatrixSumColumns(
    MatrixPack a, MatrixPack b) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < a.rows) {
    float result = 0.0f;
    for (int j = 0; j < a.cols; ++j) {
      result += a.get(i, j, 0);
    }
    b.set(i, 0, result);
  }
}

Matrix Matrix::Sum(bool layered, int layers) const {
  if (!layered) {
    assert(rows_ == layers);
    // sum columns
    assert(depth_ == 1);
    Matrix result(rows_, 1, 1);
    MatrixSumColumns<<<(rows_ + 255) / 256, 256>>>(
        MatrixPack(*this),
        MatrixPack(result));
    CUDA_ASYNC_CHECK();
    return result;
  } else {
    // sum layers
    assert(layers > 0);
    assert(depth_ % layers == 0);
    Matrix result(1, 1, layers);
    MatrixSumLayers<<<(layers + 7) / 8, 8>>>(
        MatrixPack(*this),
        MatrixPack(result));
    CUDA_ASYNC_CHECK();
    return result;
  }
}

__global__ void MatrixRepeatLayers(
    MatrixPack a,
    MatrixPack b) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int k = threadIdx.z + blockDim.z * blockIdx.z;
  if (b.inside(i, j, k)) {
    b.set(i, j, k, a.get(0, 0, k % a.depth));
  }
}

__global__ void MatrixRepeatColumns(
    MatrixPack a,
    MatrixPack b) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;

  if (b.inside(i, j)) {
    b.set(i, j, a.get(i, 0));
  }
}

Matrix Matrix::Repeat(
    bool layered, int rows, int cols, int depth) const {
  if (layered) {
    assert(depth > 0);
    assert(depth % depth_ == 0);
    assert(rows_ == 1);
    assert(cols_ == 1);
    Matrix result(rows, cols, depth);
    dim3 threads_per_block(16, 16, 1);
    dim3 blocks = CalculateBlocks(result, threads_per_block);
    MatrixRepeatLayers<<<blocks, threads_per_block>>>(
        MatrixPack(*this),
        MatrixPack(result));
    CUDA_ASYNC_CHECK();
    return result;
  } else {
    assert(rows % rows_ == 0);
    assert(depth == 1);
    assert(depth_ == 1);
    assert(cols_ == 1);
    Matrix result(rows, cols, depth);
    dim3 threads_per_block(16, 16, 1);
    dim3 blocks = CalculateBlocks(result, threads_per_block);
    MatrixRepeatColumns<<<blocks, threads_per_block>>>(
        MatrixPack(*this),
        MatrixPack(result));
    CUDA_ASYNC_CHECK();
    return result;
  }
}

Matrix Matrix::Repeat(bool layered, const Matrix& size_template) const {
  return Repeat(
      layered,
      size_template.rows(),
      size_template.cols(),
      size_template.depth());
}

__global__ void MatrixPerLayerSum(
    MatrixPack a,
    MatrixPack b) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int k = threadIdx.z + blockDim.z * blockIdx.z;

  if (b.inside(i, j, k)) {
    float sum = 0.0f;
    for (int k1 = k; k1 < a.depth; k1 += b.depth) {
      sum += a.get(i, j, k1);
    }
    b.set(i, j, k, sum);
  }
}
Matrix Matrix::PerLayerSum(int layers) const {
  assert(depth_ % layers == 0);
  Matrix result(rows_, cols_, layers);
  dim3 threads_per_block(16, 16, 1);
  dim3 blocks = CalculateBlocks(result, threads_per_block);
  MatrixPerLayerSum<<<blocks, threads_per_block>>>(
      MatrixPack(*this),
      MatrixPack(result));
  CUDA_ASYNC_CHECK();
  return result;
}

__global__ void MatrixPerLayerRepeat(
    MatrixPack a,
    MatrixPack b) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int k = threadIdx.z + blockDim.z * blockIdx.z;

  if (b.inside(i, j, k)) {
    b.set(i, j, k, a.get(i, j, k % a.depth));
  }
}

Matrix Matrix::PerLayerRepeat(int times) const {
  Matrix result(rows_, cols_, depth_ * times);
  dim3 threads_per_block(16, 16, 1);
  dim3 blocks = CalculateBlocks(result, threads_per_block);
  MatrixPerLayerRepeat<<<blocks, threads_per_block>>>(
      MatrixPack(*this),
      MatrixPack(result));
  CUDA_ASYNC_CHECK();
  return result;
}

__global__ void VecL2(float* A, int len, float* B) {
  float result = 0.0;
  for (int i = 0; i < len; ++i) {
    result += A[i] * A[i];
  }
  B[0] = result;
}

float Matrix::L2() const {
  Matrix result(1, 1, 1);
  VecL2<<<1, 1>>>(data_.get(), size_, result.data_.get());
  CUDA_ASYNC_CHECK();
  return result.GetValue(0, 0, 0);
}

__global__ void VecSoftmax(MatrixPack a, MatrixPack b, MatrixPack c) {
  int col = threadIdx.x + blockDim.x * blockIdx.x;
  if (col < a.cols) {

    // Get max value from column. Needed for numerical stability, see
    // http://cs231n.github.io/linear-classify/#softmax
    float max_val = a.get(0, col);
    for (int i = 1; i < a.rows; i++) {
      float val = a.get(i, col);
      if (val > max_val) {
        max_val = val;
      }
    }

    int expected_class = static_cast<int>(b.get(0, col));
    float expected_class_score = -1.0;
    float sum = 0.0f;
    for (int i = 0; i < a.rows; ++i) {
      float val = a.get(i, col) - max_val;
      if (i == expected_class) {
        expected_class_score = val;
      }
      sum += exp(val);
    }

    c.set(0, col, -expected_class_score + log(sum));
  }
}

float Matrix::Softmax(const Matrix& expected_class) const {
  assert(depth_ == 1);
  // rows_ = number of classes
  // cols_ = number of samples (we run the same algorithm for each sample)
  assert(expected_class.rows_ == 1);
  assert(expected_class.cols_ == cols_);
  assert(expected_class.depth_ == 1);

  Matrix result(1, cols_, 1);
  VecSoftmax<<<(cols_ + 255) / 256, 256>>>(
      MatrixPack(*this),
      MatrixPack(expected_class),
      MatrixPack(result));
  CUDA_ASYNC_CHECK();
  return result.Sum();
}


__global__ void VecSoftmaxGradient(
    MatrixPack a,
    MatrixPack b,
    MatrixPack c) {
  // TODO: clean up code duplication with VecSoftmax
  int col = threadIdx.x + blockDim.x * blockIdx.x;

  if (col < a.cols) {
    float max_val = a.get(0, col);
    for (int i = 1; i < a.rows; i++) {
      float val = a.get(i, col);
      if (val > max_val) {
        max_val = val;
      }
    }

    float sum = 0.0f;
    for (int i = 0; i < a.rows; ++i) {
      float val = exp(a.get(i, col) - max_val);
      c.set(i, col, val);
      sum += val;
    }
    int expected_class = static_cast<int>(b.get(0, col));
    for (int i = 0; i < a.rows; ++i) {
      c.div(i, col, sum);
      if (i == expected_class) {
        c.add(i, col, -1.0f);
      }
    }
  }
}

Matrix Matrix::SoftmaxGradient(const Matrix& expected_class) const {
  // Covered in cnn/error_layer_test.cc.

  assert(depth_ == 1);
  // rows_ = number of classes
  // cols_ = number of samples (we run the same algorithm for each sample)
  assert(expected_class.rows_ == 1);
  assert(expected_class.cols_ == cols_);
  assert(expected_class.depth_ == 1);

  Matrix result(rows_, cols_, 1);
  VecSoftmaxGradient<<<(cols_ + 255) / 256, 256>>>(
      MatrixPack(*this),
      MatrixPack(expected_class),
      MatrixPack(result));
  CUDA_ASYNC_CHECK();
  return result;
}

__global__ void VecNumMatches(MatrixPack a, MatrixPack b, MatrixPack c) {
  int col = threadIdx.x + blockDim.x * blockIdx.x;
  if (col < a.cols) {

    // Get max value from column.
    bool unique = true;
    float max_val = a.get(0, col);
    for (int i = 1; i < a.rows; i++) {
      float val = a.get(i, col);
      if (val > max_val) {
        max_val = val;
        unique = true;
      } else if (val == max_val) {
        unique = false;
      }
    }

    if (unique) {
      int expected_class = static_cast<int>(b.get(0, col));
      float expected_class_score = a.get(expected_class, col);
      if (expected_class_score == max_val) {
        c.set(0, col, 1.0f);
      } else {
        c.set(0, col, 0.0f);
      }
    } else {
      c.set(0, col, 0.0f);
    }
  }
}

float Matrix::NumMatches(const Matrix& expected_class) const {
  assert(depth_ == 1);
  // rows_ = number of classes
  // cols_ = number of samples (we run the same algorithm for each sample)
  assert(expected_class.rows_ == 1);
  assert(expected_class.cols_ == cols_);
  assert(expected_class.depth_ == 1);

  Matrix result(1, cols_, 1);
  VecNumMatches<<<(cols_ + 255) / 256, 256>>>(
      MatrixPack(*this),
      MatrixPack(expected_class),
      MatrixPack(result));
  CUDA_ASYNC_CHECK();
  return result.Sum();
}

__global__ void VecFill(float value, float* A, int a_size) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < a_size) {
    A[i] = value;
  }
}

void Matrix::Fill(float value) {
  VecFill<<<(size_ + 255) / 256, 256>>>(value, data_.get(), size_);
  CUDA_ASYNC_CHECK();
}

__global__ void MatrixConvolution(
    int layers_per_image,
    MatrixPack a, bool a_major, int num_a_images,
    int a_row_shift, int a_col_shift,
    MatrixPack filters, bool filters_major, int num_filters_images,
    MatrixPack b) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int k = threadIdx.z + blockDim.z * blockIdx.z;
  if (b.inside(i, j, k)) {
    // k: destination depth-level = id of filter to apply

    // layout of resulting matrix (list of layers):
    //
    // 1st image with 1st filter
    // 1st image with 2nd filter
    // ...
    // 2nd image with 1st filter
    // 2nd image with 2nd filter
    // ...

    int a_i = i + a_row_shift;
    int a_j = j + a_col_shift;

    int filter_id = k % num_filters_images;
    int image_id = k / num_filters_images;

    float sum = 0.0;
    for (int fk = 0; fk < layers_per_image; ++fk) {

      int filters_k = 0;  // Layer id in |filters| to use below.
      int a_k = 0;   // Layer id in |a| to use now below.
      if (a_major) {
        a_k = fk + image_id * layers_per_image;
      } else {
        a_k = fk * num_a_images + image_id;
      }
      if (filters_major) {
        filters_k = fk + filter_id * layers_per_image;
      } else {
        filters_k = fk * num_filters_images + filter_id;
      }

      for (int fi = 0; fi < filters.rows; ++fi) {
        for (int fj = 0; fj < filters.cols; ++fj) {
          if (fi >= -a_i && fi < a.rows - a_i && fj >= -a_j && fj < a.cols - a_j) {
            sum += filters.get(fi, fj, filters_k) * a.get(a_i + fi, a_j + fj, a_k);
          }
        }
      }
    }
    b.set(i, j, k, sum);
  }
}

Matrix Matrix::Convolution(
    int layers_per_image,
    const Matrix& a, bool a_major,
    const Matrix& b, bool b_major) {
  return Convolution(
      layers_per_image,
      a, a_major, 0, 0,
      b, b_major, 0, 0,
      0, 0);
}

Matrix Matrix::Convolution(
    int layers_per_image,
    const Matrix& a, bool a_major, int a_row_padding, int a_col_padding,
    const Matrix& b, bool b_major, int b_row_padding, int b_col_padding,
    int c_row_padding, int c_col_padding) {
  int row_slots = a.rows() + 2 * a_row_padding - b.rows() - 2 * b_row_padding + 1;
  int col_slots = a.cols() + 2 * a_col_padding - b.cols() - 2 * b_col_padding + 1;
  assert(a.depth() % layers_per_image == 0);
  assert(b.depth() % layers_per_image == 0);
  int num_a_images = a.depth() / layers_per_image;
  int num_b_images = b.depth() / layers_per_image;
  Matrix c(
      row_slots - 2 * c_row_padding,
      col_slots - 2 * c_col_padding,
      num_a_images * num_b_images);
  int a_row_shift = c_row_padding - a_row_padding + b_row_padding;
  int a_col_shift = c_col_padding - a_col_padding + b_col_padding;

  dim3 threads_per_block(1, 1, 32);
  dim3 blocks = CalculateBlocks(c, threads_per_block);
  MatrixConvolution<<<blocks, threads_per_block>>>(
      layers_per_image,
      MatrixPack(a), a_major, num_a_images,
      a_row_shift, a_col_shift,
      MatrixPack(b), b_major, num_b_images,
      MatrixPack(c));
  CUDA_ASYNC_CHECK();
  return c;
}

__global__ void MatrixPooling(
    int pool_rows, int pool_cols,
    MatrixPack a,
    MatrixPack pooled,
    MatrixPack switches) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int k = threadIdx.z + blockDim.z * blockIdx.z;
  if (i < pooled.rows && j < pooled.cols && k < pooled.depth) {
    int best_sub_index = -1;
    float best_value = 0;
    for (int a_sub_index = 0; a_sub_index < pool_rows * pool_cols; a_sub_index++) {
      float value = a.get(
          i * pool_rows + a_sub_index / pool_cols,
          j * pool_cols + a_sub_index % pool_cols,
          k);
      if (best_sub_index < 0 || value > best_value) {
        best_sub_index = a_sub_index;
        best_value = value;
      }

    }
    pooled.set(i, j, k, best_value);
    switches.set(i, j, k, best_sub_index);
  }
}

std::pair<Matrix, Matrix> Matrix::Pooling(
    int pool_rows, int pool_cols) const {
  assert(rows_ % pool_rows == 0);
  assert(cols_ % pool_cols == 0);

  Matrix pooled(rows_ / pool_rows, cols_ / pool_cols, depth_);
  Matrix switches(rows_ / pool_rows, cols_ / pool_cols, depth_);

  dim3 threads_per_block(16, 16, 1);
  dim3 blocks = CalculateBlocks(pooled, threads_per_block);
  MatrixPooling<<<blocks, threads_per_block>>>(
      pool_rows, pool_cols,
      MatrixPack(*this),
      MatrixPack(pooled),
      MatrixPack(switches));
  CUDA_ASYNC_CHECK();

  return std::make_pair(pooled, switches);
}

__global__ void MatrixPoolingSwitch(
    int pool_rows, int pool_cols,
    MatrixPack switches,
    MatrixPack input,
    MatrixPack result) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int k = threadIdx.z + blockDim.z * blockIdx.z;
  if (input.inside(i, j, k)) {
    int sub_index = switches.get(i, j, k);
    result.set(
        i * pool_rows + sub_index / pool_cols,
        j * pool_cols + sub_index % pool_cols,
        k,
        input.get(i, j, k));
  }
}

Matrix Matrix::PoolingSwitch(
    const Matrix& switches,
    int pool_rows, int pool_cols) const {
  AssertSameDimensions(switches);

  Matrix result(rows_ * pool_rows, cols_ * pool_cols, depth_);
  result.Fill(0);

  dim3 threads_per_block(16, 16, 1);
  dim3 blocks = CalculateBlocks(switches, threads_per_block);
  MatrixPoolingSwitch<<<blocks, threads_per_block>>>(
      pool_rows, pool_cols,
      MatrixPack(switches),
      MatrixPack(*this),
      MatrixPack(result));
  CUDA_ASYNC_CHECK();
  return result;
}

Matrix Matrix::ReshapeToColumns(int unit_depth) const {
  assert(depth_ % unit_depth == 0);
  Matrix rows(*this);
  rows.cols_ = rows_ * cols_ * unit_depth;
  rows.rows_ = depth_ / unit_depth;
  rows.depth_ = 1;
  return rows.T();
}

Matrix Matrix::ReshapeFromColumns(int unit_rows, int unit_cols, int unit_depth) const {
  assert(unit_rows * unit_cols * unit_depth == rows_);
  Matrix rows(this->T());
  rows.depth_ = rows.rows_ * rows.cols_ / (unit_rows * unit_cols);
  rows.rows_ = unit_rows;
  rows.cols_ = unit_cols;
  return rows;
}

__global__ void VecInvertedDropoutFill(
    float* A,
    int size,
    float p) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    A[i] = A[i] < p ? (1.0 / p) : 0.0;
  }
}

// static
Matrix Matrix::MakeInvertedDropoutMask(
    bool layered, int num_neurons, float p, Random* random) {
  unsigned long seed = random->RandLongUnsigned();
  curandGenerator_t gen;
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed));

  Matrix result(
      layered ? 1 : num_neurons,
      1,
      layered ? num_neurons : 1);
  CURAND_CALL(curandGenerateUniform(gen, result.data_.get(), result.size()));

  VecInvertedDropoutFill<<<(255 + result.size()) / 256, 256>>>(
      result.data_.get(), result.size(), p);
  CUDA_ASYNC_CHECK();

  CURAND_CALL(curandDestroyGenerator(gen));

  return result;
}

Matrix Matrix::DeepCopy() const {
  Matrix result(rows_, cols_, depth_);
  CUDA_CALL(cudaMemcpy(
      result.data_.get(),
      data_.get(),
      size_ * sizeof(float),
      cudaMemcpyDeviceToDevice));
  return result;
}

float Matrix::GetValue(int row, int col, int depth) const {
  float result;
  CUDA_CALL(cudaMemcpy(
      &result,
      data_.get() + Index(row, col, depth),
      sizeof(float),
      cudaMemcpyDeviceToHost));
  return result;
}

void Matrix::SetValue(int row, int col, int depth, float value) {
  CUDA_CALL(cudaMemcpy(
      data_.get() + Index(row, col, depth),
      &value,
      sizeof(float),
      cudaMemcpyHostToDevice));
}

void Matrix::save(cereal::PortableBinaryOutputArchive& ar) const {
  std::vector<float> values = GetVector();
  ar(rows_, cols_, depth_, values);
}

void Matrix::load(cereal::PortableBinaryInputArchive& ar) {
  std::vector<float> values;
  ar(rows_, cols_, depth_, values);
  size_ = rows_ * cols_ * depth_;
  SetVector(values);
}
