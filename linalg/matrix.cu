#include "linalg/matrix.h"

#include <cassert>  // TODO: release-mode assert
#include <iostream>
#include <iomanip>
#include <math.h>

#include <cuda_runtime.h>

__device__ int Dim3toDim1(
    int i, int j, int k,
    int rows, int cols, int depth) {
  return k * rows * cols + i * cols + j;
}

int Matrix::Index(int i, int j, int k) const {
  return k * rows_ * cols_ + i * cols_ + j;
}

Matrix::Matrix() :
    rows_(0),
    cols_(0),
    depth_(0),
    size_(0),
    data_(NULL) {}

std::shared_ptr<float> AllocateData(int size) {
  float* data;
  cudaMalloc(&data, size * sizeof(float));
  return std::shared_ptr<float>(data, cudaFree);
}

std::shared_ptr<float> ImportData(float size, const float* host_data) {
  std::shared_ptr<float> device_data(AllocateData(size));
  cudaMemcpy(
      device_data.get(),
      host_data,
      size * sizeof(float),
      cudaMemcpyHostToDevice);
  return device_data;
}

Matrix::Matrix(int rows, int cols, int depth, float* data) :
    rows_(rows),
    cols_(cols),
    depth_(depth),
    size_(rows * cols * depth) {
  data_ = ImportData(size_, data);
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
  cudaMemcpy(
      host_data.get(),
      data_.get(),
      size_ * sizeof(float),
      cudaMemcpyDeviceToHost);
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
  return result;
}

__global__ void MatrixTranspose(float* A, int rows_, int cols_, float* T) {
  int a_index = threadIdx.x * cols_ + threadIdx.y;
  int t_index = threadIdx.y * rows_ + threadIdx.x;
  T[t_index] = A[a_index];
}

Matrix Matrix::T() const {
  assert(depth_ == 1);
  Matrix result(cols_, rows_, depth_);

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

Matrix Matrix::Rot180() const {
  Matrix result(rows_, cols_, depth_);

  dim3 grid(1, 1, 1);
  dim3 threads(rows_, cols_, depth_);
  MatrixRot180<<<grid, threads>>>(
      data_.get(),
      rows_, cols_, depth_,
      result.data_.get());
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

void GetConfigForMatrix(
    int rows, int cols, int depth,
    dim3* threadsPerBlock,
    dim3* blocks) {
  assert(depth == 1);
  *threadsPerBlock = dim3(16, 16);
  *blocks = dim3((rows + 15) / 16, (cols + 15) / 16);
}

Matrix Matrix::Dot(const Matrix& other) const {
  assert(cols_ == other.rows_);
  assert(depth_ == 1);
  int c_rows = rows_;
  int c_cols = other.cols_;
  Matrix result(c_rows, c_cols, 1);

  dim3 threadsPerBlock, blocks;
  GetConfigForMatrix(c_rows, c_cols, 1, &threadsPerBlock, &blocks);
  MatrixDotProd<<<blocks, threadsPerBlock>>>(
      data_.get(), rows_, cols_,
      other.data_.get(), other.rows_, other.cols_,
      result.data_.get(), result.rows_, result.cols_);
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
    dim3 blocks((rows + 15) / 16, (cols + 15) / 16, depth);
    MatrixRepeatLayers<<<blocks, threads_per_block>>>(
        data_.get(), depth_,
        result.data_.get(), rows, cols, depth);
    return result;
  } else {
    assert(rows % rows_ == 0);
    assert(depth == 1);
    assert(depth_ == 1);
    assert(cols_ == 1);
    Matrix result(rows, cols, depth);
    dim3 threads_per_block(16, 16, 1);
    dim3 blocks((rows + 15) / 16, (cols + 15) / 16, 1);
    MatrixRepeatColumns<<<blocks, threads_per_block>>>(
        data_.get(),
        rows, cols,
        result.data_.get());

    return result;
  }
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
}

__global__ void MatrixPadding(
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

  dim3 threadsPerBlock(16, 16, 1);
  dim3 blocks((rows_ + 15) / 16, (cols_ + 15) / 16, depth_);
  MatrixPadding<<<blocks, threadsPerBlock>>>(
      data_.get(), rows_, cols_, depth_,
      row_padding, col_padding,
      result.data_.get());
  return result;
}

__global__ void MatrixConvolution(
    int layers_per_image,
    float* A, int a_rows, int a_cols, int a_depth,
    float* filters, int f_rows, int f_cols, int f_depth,
    float* B, int b_rows, int b_cols, int b_depth) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int k = threadIdx.z + blockDim.z * blockIdx.z;
  if (i < b_rows && j < b_cols && k < b_depth) {
    // k: destination depth-level = id of filter to apply

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
}

Matrix Matrix::Convolution(
    const Matrix& filters,
    int layers_per_image,
    int stride) const {
  int row_slots = rows_ - filters.rows() + 1;
  int col_slots = cols_ - filters.cols() + 1;
  assert(row_slots % stride == 0 && col_slots % stride == 0);

  assert(filters.depth() % layers_per_image == 0);
  assert(depth() % layers_per_image == 0);

  assert(stride == 1);  // TODO
  Matrix result(
      row_slots / stride,
      col_slots / stride,
      filters.depth() / layers_per_image * depth() / layers_per_image);
  dim3 threadsPerBlock(16, 16, 1);
  dim3 blocks((result.rows() + 15) / 16, (result.cols() + 15) / 16, result.depth());
  MatrixConvolution<<<blocks, threadsPerBlock>>>(
      layers_per_image,
      data_.get(), rows_, cols_, depth_,
      filters.data_.get(), filters.rows(), filters.cols(), filters.depth(),
      result.data_.get(), result.rows(), result.cols(), result.depth());

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

Matrix Matrix::ReorderLayers(int layers_per_image) const {
  assert(depth_ % layers_per_image == 0);
  Matrix result(rows_, cols_, depth_);
  int layer_size = rows_ * cols_;
  int num_images = depth_ / layers_per_image;
  for (int src = 0; src < depth_; ++src) {
    int image_id = src / layers_per_image;
    int sublayer_id = src % layers_per_image;
    int dst = sublayer_id * num_images + image_id;
    cudaMemcpy(
        result.data_.get() + dst * layer_size,
        data_.get() + src * layer_size,
        layer_size * sizeof(float),
        cudaMemcpyDeviceToDevice);
  }

  return result;
}

Matrix Matrix::DeepCopy() const {
  Matrix result(rows_, cols_, depth_);
  cudaMemcpy(
      result.data_.get(),
      data_.get(),
      size_ * sizeof(float),
      cudaMemcpyDeviceToDevice);
  return result;
}

float Matrix::GetValue(int row, int col, int depth) const {
  float result;
  cudaMemcpy(
      &result,
      data_.get() + Index(row, col, depth),
      sizeof(float),
      cudaMemcpyDeviceToHost);
  return result;
}

void Matrix::SetValue(int row, int col, int depth, float value) {
  cudaMemcpy(
      data_.get() + Index(row, col, depth),
      &value,
      sizeof(float),
      cudaMemcpyHostToDevice);
}

