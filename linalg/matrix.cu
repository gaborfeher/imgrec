#include "linalg/matrix.h"

#include "linalg/cuda_util.h"

#include <cassert>  // TODO: release-mode assert
#include <iostream>
#include <iomanip>
#include <math.h>

#include <cuda.h>  // strangely, not needed by nvcc
#include <curand.h>

__device__ int Dim3toDim1(
    int i, int j, int k,
    int rows, int cols, int depth) {
  return k * rows * cols + i * cols + j;
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
  int row_padding;
  int col_padding;

  explicit MatrixPack(const Matrix& m, int row_padding, int col_padding) :
      items(m.data_.get()),
      rows(m.rows()),
      cols(m.cols()),
      depth(m.depth()),
      layer_size(m.rows() * m.cols()),
      row_padding(row_padding),
      col_padding(col_padding) {}

  __forceinline__ __device__ float get(int i, int j, int k) {
    // i -= row_padding; j -= col_padding;
    return items[k * layer_size + i * cols + j];
  }

  __forceinline__ __device__ void set(int i, int j, int k, float f) {
    // i -= row_padding; j -= col_padding;
    items[k * layer_size + i * cols + j] = f;
  }

  __forceinline__ __device__ bool inside(int i, int j, int k) {
    // i -= row_padding; j -= col_padding;
    return /* i >= 0 && j >= 0 &&*/ i < rows && j < cols && k < depth;
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

std::shared_ptr<float> ImportData(float size, const float* host_data) {
  std::shared_ptr<float> device_data(AllocateData(size));
  CUDA_CALL(cudaMemcpy(
      device_data.get(),
      host_data,
      size * sizeof(float),
      cudaMemcpyHostToDevice));
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
  Fill(0);
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
  std::cout << std::fixed << std::setw( 6 ) << std::setprecision( 4 );
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
    break;
  }
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

__global__ void VecAdd(float* A, float* B, float* C, int size) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    C[i] = A[i] + B[i];
  }
}

Matrix Matrix::Add(const Matrix& other) const {
  AssertSameDimensions(other);
  Matrix result(rows_, cols_, depth_);
  VecAdd<<<(size_ + 255) / 256, 256>>>(data_.get(), other.data_.get(), result.data_.get(), size_);
  CUDA_ASYNC_CHECK();
  return result;
}

__global__ void VecAddConst(float* A, float b, float* B, int size) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    B[i] = A[i] + b;
  }
}

Matrix Matrix::AddConst(float c) const {
  Matrix result(rows_, cols_, depth_);
  VecAddConst<<<(size_ + 255) / 256, 256>>>(data_.get(), c, result.data_.get(), size_);
  CUDA_ASYNC_CHECK();
  return result;
}

__global__ void VecPow(float* A, float exp, float* B, int size) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    B[i] = pow(A[i], exp);
  }
}

Matrix Matrix::Pow(float exp) const {
  Matrix result(rows_, cols_, depth_);
  VecPow<<<(size_ + 255) / 256, 256>>>(data_.get(), exp, result.data_.get(), size_);
  CUDA_ASYNC_CHECK();
  return result;
}

__global__ void VecMult(float* A, float* B, float* C, int size) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    C[i] = A[i] * B[i];
  }
}

Matrix Matrix::ElementwiseMultiply(const Matrix& other) const {
  AssertSameDimensions(other);
  Matrix result(rows_, cols_, depth_);
  VecMult<<<(size_ + 255) / 256, 256>>>(data_.get(), other.data_.get(), result.data_.get(), size_);
  CUDA_ASYNC_CHECK();
  return result;
}

__global__ void VecDivide(float* A, float* B, float* C, int size) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    C[i] = A[i] / B[i];
  }
}

Matrix Matrix::ElementwiseDivide(const Matrix& other) const {
  AssertSameDimensions(other);
  Matrix result(rows_, cols_, depth_);
  VecDivide<<<(size_ + 255) / 256, 256>>>(data_.get(), other.data_.get(), result.data_.get(), size_);
  CUDA_ASYNC_CHECK();
  return result;
}

__global__ void MatrixTranspose(float* A, int rows, int cols, float* T) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  if (i < rows && j < cols) {
    int a_index = i * cols + j;
    int t_index = j * rows + i;
    T[t_index] = A[a_index];
  }
}

Matrix Matrix::T() const {
  assert(depth_ == 1);
  Matrix result(cols_, rows_, depth_);

  dim3 threads_per_block(16, 16, 1);
  dim3 blocks = CalculateBlocks(*this, threads_per_block);
  MatrixTranspose<<<blocks, threads_per_block>>>(
      data_.get(), rows_, cols_, result.data_.get());
  CUDA_ASYNC_CHECK();
  return result;
}

__global__ void MatrixRot180(
    float* A,
    int rows, int cols, int depth,
    float* R) {

  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int k = threadIdx.z + blockDim.z * blockIdx.z;
  if (i < rows && j < cols && k < depth) {
    int a_index = Dim3toDim1(
        i, j, k,
        rows, cols, depth);
    int r_index = Dim3toDim1(
        rows - i - 1, cols - j - 1, k,
        rows, cols, depth);
    R[r_index] = A[a_index];
  }
}

Matrix Matrix::Rot180() const {
  Matrix result(rows_, cols_, depth_);

  dim3 threads_per_block(16, 16, 1);
  dim3 blocks = CalculateBlocks(result, threads_per_block);
  MatrixRot180<<<blocks, threads_per_block>>>(
      data_.get(),
      rows_, cols_, depth_,
      result.data_.get());
  CUDA_ASYNC_CHECK();
  return result;
}

__global__ void VecMultiply(float* A, float m, float* B, int size) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    B[i] = A[i] * m;
  }
}

Matrix Matrix::Multiply(float m) const {
  Matrix result(rows_, cols_, depth_);
  VecMultiply<<<(size_ + 255) / 256, 256>>>(data_.get(), m, result.data_.get(), size_);
  CUDA_ASYNC_CHECK();
  return result;
}

__global__ void VecDivide(float* A, float d, float* B, int size) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    B[i] = A[i] / d;
  }
}

Matrix Matrix::Divide(float d) const {
  Matrix result(rows_, cols_, depth_);
  VecDivide<<<(size_ + 255) / 256, 256>>>(data_.get(), d, result.data_.get(), size_);
  CUDA_ASYNC_CHECK();
  return result;
}

__global__ void MatrixDotProd(
    float* A, int a_rows, int a_cols,
    float* B, int b_rows, int b_cols,
    float* C, int c_rows, int c_cols) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  if (i < c_rows && j < c_cols) {
    float sum = 0.0;
    for (int k = 0; k < a_cols; ++k) {
      sum += A[i * a_cols + k] * B[k * b_cols + j];
    }
    C[i * c_cols + j] = sum;
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
      data_.get(), rows_, cols_,
      other.data_.get(), other.rows_, other.cols_,
      result.data_.get(), result.rows_, result.cols_);
  CUDA_ASYNC_CHECK();
  return result;
}

__global__ void VecSigmoid(float* A, float* B, int size) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    B[i] = 1.0 / (1.0 + exp(-A[i]));
  }
}


__global__ void VecSigmoidGradient(float* A, float* B, int size) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    float sigma = 1.0 / (1.0 + exp(-A[i]));
    B[i] = sigma * (1.0 - sigma);
  }
}

__global__ void VecReLU(float* A, float* B, int size) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    B[i] = max(0.0f, A[i]);
  }
}

__global__ void VecReLUGradient(float* A, float* B, int size) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    if (A[i] < 0.0f) {
      B[i] = 0.0f;
    } else {
      B[i] = 1.0f;
    }
  }
}

__global__ void VecLReLU(float* A, float* B, int size) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    B[i] = max(0.01f * A[i], A[i]);
  }
}

__global__ void VecLReLUGradient(float* A, float* B, int size) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    if (A[i] < 0.0f) {
      B[i] = 0.01f;
    } else {
      B[i] = 1.0f;
    }
  }
}

__global__ void VecSquare(float* A, float* B, int size) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    B[i] = A[i] * A[i];
  }
}

__global__ void VecSqrt(float* A, float* B, int size) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    B[i] = sqrt(A[i]);
  }
}

namespace matrix_mappers {

// We provide factory methdos instead of direct implementations
// so that users of device_matrix.h won't need to depend on
// CUDA stuff.

MapperFunc Sigmoid() {
  return &VecSigmoid;
}

MapperFunc SigmoidGradient() {
  return &VecSigmoidGradient;
}

MapperFunc ReLU() {
  return &VecReLU;
}

MapperFunc ReLUGradient() {
  return &VecReLUGradient;
}

MapperFunc LReLU() {
  return &VecLReLU;
}

MapperFunc LReLUGradient() {
  return &VecLReLUGradient;
}

MapperFunc Square() {
  return &VecSquare;
}

MapperFunc Sqrt() {
  return &VecSqrt;
}

}  // namespacce matrix_mappers

Matrix Matrix::Map(::matrix_mappers::MapperFunc map) const {
  Matrix result(rows_, cols_, depth_);
  map<<<(size_ + 255) / 256, 256>>>(
      data_.get(),
      result.data_.get(),
      size_);
  CUDA_ASYNC_CHECK();
  return result;
}

__global__ void VecSum(float* A, int len, float* B) {
  float result = 0.0;
  for (int i = 0; i < len; ++i) {
    result += A[i];
  }
  B[0] = result;
}

float Matrix::Sum() const {
  Matrix result(1, 1, 1);
  VecSum<<<1, 1>>>(data_.get(), size_, result.data_.get());
  CUDA_ASYNC_CHECK();
  return result.GetValue(0, 0, 0);
}

__global__ void MatrixSumLayers(
    float* A,
    int a_rows, int a_cols, int a_depth,
    float* B,
    int b_depth) {
  int b_index = threadIdx.x + blockDim.x * blockIdx.x;
  if (b_index < b_depth) {
    float result = 0.0;
    for (int i = 0; i < a_rows; ++i) {
      for (int j = 0; j < a_cols; ++j) {
        for (int k = b_index; k < a_depth; k += b_depth) {
          result += A[Dim3toDim1(i, j, k, a_rows, a_cols, a_depth)];
        }
      }
    }
    B[b_index] = result;
  }
}

__global__ void MatrixSumColumns(
    float* A,
    int rows, int cols,
    float* B) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < rows) {
    float result = 0.0f;
    for (int j = 0; j < cols; ++j) {
      result += A[Dim3toDim1(i, j, 0, rows, cols, 1)];
    }
    B[i] = result;
  }
}

Matrix Matrix::Sum(bool layered, int layers) const {
  if (!layered) {
    assert(rows_ == layers);
    // sum columns
    assert(depth_ == 1);
    Matrix result(rows_, 1, 1);
    MatrixSumColumns<<<(rows_ + 255) / 256, 256>>>(
        data_.get(),
        rows_, cols_,
        result.data_.get());
    CUDA_ASYNC_CHECK();
    return result;
  } else {
    // sum layers
    assert(layers > 0);
    assert(depth_ % layers == 0);
    Matrix result(1, 1, layers);
    MatrixSumLayers<<<(layers + 255) / 256, 256>>>(
        data_.get(),
        rows_, cols_, depth_,
        result.data_.get(),
        layers);
    CUDA_ASYNC_CHECK();
    return result;
  }
}

__global__ void MatrixRepeatLayers(
    float* A, int a_depth,
    float* B, int b_rows, int b_cols, int b_depth) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int k = threadIdx.z + blockDim.z * blockIdx.z;

  if (i < b_rows && j < b_cols && k < b_depth) {
    int b_index = Dim3toDim1(
        i, j, k,
        b_rows, b_cols, b_depth);
    int a_index = Dim3toDim1(0, 0, k % a_depth, 1, 1, a_depth);
    B[b_index] = A[a_index];
  }
}

__global__ void MatrixRepeatColumns(
    float* A,
    int rows, int cols,
    float* B) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;

  if (i < rows && j < cols) {
    int b_index = Dim3toDim1(i, j, 0, rows, cols, 1);
    B[b_index] = A[i];
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
        data_.get(), depth_,
        result.data_.get(), rows, cols, depth);
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
        data_.get(),
        rows, cols,
        result.data_.get());
    CUDA_ASYNC_CHECK();
    return result;
  }
}

__global__ void MatrixPerLayerSum(
    float* A,
    int rows, int cols, int a_depth,
    float* B,
    int b_depth) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int k = threadIdx.z + blockDim.z * blockIdx.z;

  if (i < rows && j < cols && k < b_depth) {
    float sum = 0.0f;
    for (int k1 = k; k1 < a_depth; k1 += b_depth) {
      int a_index = Dim3toDim1(i, j, k1, rows, cols, a_depth);
      sum += A[a_index];
    }
    int b_index = Dim3toDim1(i, j, k, rows, cols, b_depth);
    B[b_index] = sum;
  }
}
Matrix Matrix::PerLayerSum(int layers) const {
  assert(depth_ % layers == 0);
  Matrix result(rows_, cols_, layers);
  dim3 threads_per_block(16, 16, 1);
  dim3 blocks = CalculateBlocks(result, threads_per_block);
  MatrixPerLayerSum<<<blocks, threads_per_block>>>(
        data_.get(), rows_, cols_, depth_,
        result.data_.get(), layers);
  CUDA_ASYNC_CHECK();
  return result;
}

__global__ void MatrixPerLayerRepeat(
    float* A,
    int rows, int cols, int a_depth,
    float* B,
    int b_depth) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int k = threadIdx.z + blockDim.z * blockIdx.z;

  if (i < rows && j < cols && k < b_depth) {
    int a_index = Dim3toDim1(i, j, k % a_depth, rows, cols, a_depth);
    int b_index = Dim3toDim1(i, j, k, rows, cols, b_depth);
    B[b_index] = A[a_index];
  }
}
Matrix Matrix::PerLayerRepeat(int times) const {
  Matrix result(rows_, cols_, depth_ * times);
  dim3 threads_per_block(16, 16, 1);
  dim3 blocks = CalculateBlocks(result, threads_per_block);
  MatrixPerLayerRepeat<<<blocks, threads_per_block>>>(
        data_.get(), rows_, cols_, depth_,
        result.data_.get(), result.depth());
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
  // TODO: use the following, but figure out while it fails the tests now:
  // return Map(::matrix_mappers::Square()).Sum();
}


__global__ void VecSoftmax(float* A, int a_rows, int a_cols, float* B, float* C) {
  int col = threadIdx.x + blockDim.x * blockIdx.x;
  if (col < a_cols) {

    // Get max value from column. Needed for numerical stability, see
    // http://cs231n.github.io/linear-classify/#softmax
    float max_val = A[Dim3toDim1(0, col, 0, a_rows, a_cols, 1)];
    for (int i = 1; i < a_rows; i++) {
      float val = A[Dim3toDim1(i, col, 0, a_rows, a_cols, 1)];
      if (val > max_val) {
        max_val = val;
      }
    }

    int expected_class = static_cast<int>(B[col]);
    float expected_class_score = -1.0;
    float sum = 0.0f;
    for (int i = 0; i < a_rows; ++i) {
      float val = A[Dim3toDim1(i, col, 0, a_rows, a_cols, 1)] - max_val;
      if (i == expected_class) {
        expected_class_score = val;
      }
      sum += exp(val);
    }

    C[col] = -expected_class_score + log(sum);
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
      data_.get(), rows_, cols_,
      expected_class.data_.get(),
      result.data_.get());
  CUDA_ASYNC_CHECK();
  return result.Sum();
}


__global__ void VecSoftmaxGradient(float* A, int a_rows, int a_cols, float* B, float* C) {
  // TODO: clean up code duplication with VecSoftmax
  int col = threadIdx.x + blockDim.x * blockIdx.x;

  if (col < a_cols) {

    float max_val = A[Dim3toDim1(0, col, 0, a_rows, a_cols, 1)];
    for (int i = 1; i < a_rows; i++) {
      int index = Dim3toDim1(i, col, 0, a_rows, a_cols, 1);
      float val = A[index];
      if (val > max_val) {
        max_val = val;
      }
    }

    float sum = 0.0f;
    for (int i = 0; i < a_rows; ++i) {
      int index = Dim3toDim1(i, col, 0, a_rows, a_cols, 1);
      float val = exp(A[index] - max_val);
      C[index] = val;
      sum += val;
    }
    int expected_class = static_cast<int>(B[col]);
    for (int i = 0; i < a_rows; ++i) {
      int index = Dim3toDim1(i, col, 0, a_rows, a_cols, 1);
      C[index] = C[index] / sum;
      if (i == expected_class) {
        C[index] -= 1.0f;
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
      data_.get(), rows_, cols_,
      expected_class.data_.get(),
      result.data_.get());
  CUDA_ASYNC_CHECK();
  return result;
}

__global__ void VecNumMatches(float* A, int a_rows, int a_cols, float* B, float* C) {
  int col = threadIdx.x + blockDim.x * blockIdx.x;
  if (col < a_cols) {

    // Get max value from column.
    bool unique = true;
    float max_val = A[Dim3toDim1(0, col, 0, a_rows, a_cols, 1)];
    for (int i = 1; i < a_rows; i++) {
      float val = A[Dim3toDim1(i, col, 0, a_rows, a_cols, 1)];
      if (val > max_val) {
        max_val = val;
        unique = true;
      } else if (val == max_val) {
        unique = false;
      }
    }

    if (unique) {
      int expected_class = static_cast<int>(B[col]);
      float expected_class_score = A[Dim3toDim1(expected_class, col, 0, a_rows, a_cols, 1)];
      if (expected_class_score == max_val) {
        C[col] = 1.0f;
      } else {
        C[col] = 0.0f;
      }
    } else {
      C[col] = 0.0f;
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
      data_.get(), rows_, cols_,
      expected_class.data_.get(),
      result.data_.get());
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

__global__ void MatrixAddPadding(
    float* A,
    int rows, int cols, int depth,
    int row_padding, int col_padding,
    float* B) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int k = threadIdx.z + blockDim.z * blockIdx.z;

  if (i < rows && j < cols && k < depth) {
    int b_index = Dim3toDim1(
        i + row_padding, j + col_padding, k,
        rows + 2 * row_padding,
        cols + 2 * col_padding,
        depth);
    int a_index = Dim3toDim1(i, j, k, rows, cols, depth);
    B[b_index] = A[a_index];
  }
}

Matrix Matrix::AddPadding(
    int row_padding, int col_padding) const {
  if (row_padding <= 0 && col_padding <= 0) {
    return *this;
  }

  Matrix result(
      rows_ + 2 * row_padding,
      cols_ + 2 * col_padding,
      depth_);  // filled with zeros

  dim3 threads_per_block(16, 16, 1);
  dim3 blocks = CalculateBlocks(*this, threads_per_block);
  MatrixAddPadding<<<blocks, threads_per_block>>>(
      data_.get(), rows_, cols_, depth_,
      row_padding, col_padding,
      result.data_.get());
  CUDA_ASYNC_CHECK();
  return result;
}

__global__ void MatrixRemovePadding(
    float* A,
    int rows, int cols, int depth,
    int row_padding, int col_padding,
    float* B) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int k = threadIdx.z + blockDim.z * blockIdx.z;

  if (i < rows && j < cols && k < depth) {
    int a_index = Dim3toDim1(
        i + row_padding, j + col_padding, k,
        rows + 2 * row_padding,
        cols + 2 * col_padding,
        depth);
    int b_index = Dim3toDim1(i, j, k, rows, cols, depth);
    B[b_index] = A[a_index];
  }
}

Matrix Matrix::RemovePadding(
    int row_padding, int col_padding) const {
  if (row_padding == 0 && col_padding == 0) {
    return *this;
  }
  assert(row_padding >= 0);
  assert(col_padding >= 0);
  assert(rows_ - 2 * row_padding > 0);
  assert(cols_ - 2 * col_padding > 0);

  Matrix result(
      rows_ - 2 * row_padding,
      cols_ - 2 * col_padding,
      depth_);

  dim3 threads_per_block(16, 16, 1);
  dim3 blocks = CalculateBlocks(result, threads_per_block);
  MatrixRemovePadding<<<blocks, threads_per_block>>>(
      data_.get(), result.rows(), result.cols(), depth_,
      row_padding, col_padding,
      result.data_.get());
  CUDA_ASYNC_CHECK();
  return result;
}

__global__ void MatrixConvolution(
    int layers_per_image,
    MatrixPack a, bool a_major, int num_a_images,
    MatrixPack filters, bool filters_major, int num_filters_images,
    MatrixPack b) {

  int i = threadIdx.x + blockDim.x * blockIdx.x; // + b.row_padding;
  int j = threadIdx.y + blockDim.y * blockIdx.y; // + b.col_padding;
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

    int filter_id = k % num_filters_images;
    int image_id = k / num_filters_images;

    float sum = 0.0;
    for (int fk = 0; fk < layers_per_image; ++fk) {
      // i + f_row_start + b.row_padding - a.row_padding >= 0
      int f_row_start = max(0, a.row_padding - b.row_padding - i);
      int f_col_start = max(0, a.col_padding - b.col_padding - j);
      // i + f_row_stop + b.row_padding - a.row_padding <= a.rows
      int f_row_stop = min(filters.rows, a.rows + a.row_padding - b.row_padding - i);
      int f_col_stop = min(filters.cols, a.cols + a.col_padding - b.col_padding - j);

      for (int fi = f_row_start; fi < f_row_stop; ++fi) {
        for (int fj = f_col_start; fj < f_col_stop; ++fj) {
          int filters_k = 0;
          int a_k = 0;
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

          float f_val = filters.get(
              fi,
              fj,
              filters_k);
          float a_val = a.get(
              i + fi + b.row_padding - a.row_padding,
              j + fj + b.col_padding - a.col_padding,
              a_k);
          sum += f_val * a_val;
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
  Matrix c(row_slots - 2 * c_row_padding, col_slots - 2 * c_col_padding, num_a_images * num_b_images);

  dim3 threads_per_block(8, 8, 1);
  dim3 blocks = CalculateBlocks(c, threads_per_block);
  MatrixConvolution<<<blocks, threads_per_block>>>(
      layers_per_image,
      MatrixPack(a, a_row_padding, a_col_padding), a_major, num_a_images,
      MatrixPack(b, b_row_padding, b_col_padding), b_major, num_b_images,
      MatrixPack(c, c_row_padding, c_col_padding));
  CUDA_ASYNC_CHECK();
  return c;
}

__global__ void MatrixPooling(
    int pool_rows, int pool_cols,
    float* A,
    int a_rows, int a_cols, int a_depth,
    float* pooled,
    float* switches,
    int pooled_rows, int pooled_cols, int pooled_depth) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int k = threadIdx.z + blockDim.z * blockIdx.z;
  if (i < pooled_rows && j < pooled_cols && k < pooled_depth) {
    int best_sub_index = -1;
    float best_value = 0;
    for (int a_sub_index = 0; a_sub_index < pool_rows * pool_cols; a_sub_index++) {
      float value = A[Dim3toDim1(
          i * pool_rows + a_sub_index / pool_cols,
          j * pool_cols + a_sub_index % pool_cols,
          k,
          a_rows, a_cols, a_depth)];
      if (best_sub_index < 0 || value > best_value) {
        best_sub_index = a_sub_index;
        best_value = value;
      }

    }

    int pooled_index = Dim3toDim1(
        i, j, k, pooled_rows, pooled_cols, pooled_depth);
    pooled[pooled_index] = best_value;
    switches[pooled_index] = best_sub_index;
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
      data_.get(),
      rows_, cols_, depth_,
      pooled.data_.get(),
      switches.data_.get(),
      pooled.rows_, pooled.cols_, pooled.depth_);
  CUDA_ASYNC_CHECK();

  return std::make_pair(pooled, switches);
}

__global__ void MatrixPoolingSwitch(
    int pool_rows, int pool_cols,
    float* switches,
    float* input,
    int rows, int cols, int depth,
    float* result) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int k = threadIdx.z + blockDim.z * blockIdx.z;
  if (i < rows && j < cols && k < depth) {
    int input_index = Dim3toDim1(i, j, k, rows, cols, depth);
    int sub_index = switches[input_index];
    int result_index = Dim3toDim1(
        i * pool_rows + sub_index / pool_cols,
        j * pool_cols + sub_index % pool_cols,
        k,
        rows * pool_rows,
        cols * pool_cols,
        depth);
    result[result_index] = input[input_index];
  }
}

Matrix Matrix::PoolingSwitch(
    const Matrix& switches,
    int pool_rows, int pool_cols) const {
  AssertSameDimensions(switches);

  Matrix result(rows_ * pool_rows, cols_ * pool_cols, depth_);  // Zero-filled.

  dim3 threads_per_block(16, 16, 1);
  dim3 blocks = CalculateBlocks(switches, threads_per_block);
  MatrixPoolingSwitch<<<blocks, threads_per_block>>>(
      pool_rows, pool_cols,
      switches.data_.get(),
      data_.get(),
      rows_, cols_, depth_,
      result.data_.get());
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

void Matrix::InvertedDropoutFill(Random* random, float p) {
  unsigned long seed = random->RandLongUnsigned();

  curandGenerator_t gen;
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed));
  CURAND_CALL(curandGenerateUniform(gen, data_.get(), size_));

  VecInvertedDropoutFill<<<(255 + size_) / 256, 256>>>(
      data_.get(), size_, p);
  CUDA_ASYNC_CHECK();
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

