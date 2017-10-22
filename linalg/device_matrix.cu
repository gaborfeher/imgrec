#include "linalg/device_matrix.h"

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

int DeviceMatrix::Index(int i, int j, int k) const {
  return k * rows_ * cols_ + i * cols_ + j;
}

DeviceMatrix::DeviceMatrix() :
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

DeviceMatrix::DeviceMatrix(int rows, int cols, int depth, float* data) :
    rows_(rows),
    cols_(cols),
    depth_(depth),
    size_(rows * cols * depth) {
  data_ = ImportData(size_, data);
}

DeviceMatrix::DeviceMatrix(int rows, int cols, int depth, const std::vector<float>& data) :
    rows_(rows),
    cols_(cols),
    depth_(depth),
    size_(rows * cols * depth) {
  SetVector(data);
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

void DeviceMatrix::SetVector(const std::vector<float>& data) {
  assert(data.size() == size_);
  data_ = ImportData(size_, &data[0]);
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

void DeviceMatrix::AssertDimensions(int rows, int cols, int depth) const {
  assert(rows_ == rows && cols_ == cols && depth_ == depth);
}

void DeviceMatrix::AssertSameDimensions(const DeviceMatrix& other) const {
  assert(rows_ == other.rows_ && cols_ == other.cols_ && depth_ == other.depth_);
}

void DeviceMatrix::AssertRows(int rows) const {
  assert(rows_ == rows);
}

void DeviceMatrix::AssertDepth(int depth) const {
  assert(depth_ == depth);
}

__global__ void VecAdd(float* A, float* B, float* C, int size) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    C[i] = A[i] + B[i];
  }
}

DeviceMatrix DeviceMatrix::Add(const DeviceMatrix& other) const {
  AssertSameDimensions(other);
  DeviceMatrix result(rows_, cols_, depth_);
  VecAdd<<<(size_ + 255) / 256, 256>>>(data_.get(), other.data_.get(), result.data_.get(), size_);
  return result;
}

__global__ void VecAddConst(float* A, float b, float* B, int size) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    B[i] = A[i] + b;
  }
}

DeviceMatrix DeviceMatrix::AddConst(float c) const {
  DeviceMatrix result(rows_, cols_, depth_);
  VecAddConst<<<(size_ + 255) / 256, 256>>>(data_.get(), c, result.data_.get(), size_);
  return result;
}

__global__ void VecPow(float* A, float exp, float* B, int size) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    B[i] = pow(A[i], exp);
  }
}

DeviceMatrix DeviceMatrix::Pow(float exp) const {
  DeviceMatrix result(rows_, cols_, depth_);
  VecPow<<<(size_ + 255) / 256, 256>>>(data_.get(), exp, result.data_.get(), size_);
  return result;
}

__global__ void VecMult(float* A, float* B, float* C, int size) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    C[i] = A[i] * B[i];
  }
}

DeviceMatrix DeviceMatrix::ElementwiseMultiply(const DeviceMatrix& other) const {
  AssertSameDimensions(other);
  DeviceMatrix result(rows_, cols_, depth_);
  VecMult<<<(size_ + 255) / 256, 256>>>(data_.get(), other.data_.get(), result.data_.get(), size_);
  return result;
}

__global__ void VecDivide(float* A, float* B, float* C, int size) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    C[i] = A[i] / B[i];
  }
}

DeviceMatrix DeviceMatrix::ElementwiseDivide(const DeviceMatrix& other) const {
  AssertSameDimensions(other);
  DeviceMatrix result(rows_, cols_, depth_);
  VecDivide<<<(size_ + 255) / 256, 256>>>(data_.get(), other.data_.get(), result.data_.get(), size_);
  return result;
}

__global__ void MatrixTranspose(float* A, int rows_, int cols_, float* T) {
  int a_index = threadIdx.x * cols_ + threadIdx.y;
  int t_index = threadIdx.y * rows_ + threadIdx.x;
  T[t_index] = A[a_index];
}

DeviceMatrix DeviceMatrix::T() const {
  assert(depth_ == 1);
  DeviceMatrix result(cols_, rows_, depth_);

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
  DeviceMatrix result(rows_, cols_, depth_);

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

DeviceMatrix DeviceMatrix::Multiply(float m) const {
  DeviceMatrix result(rows_, cols_, depth_);
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

DeviceMatrix DeviceMatrix::Dot(const DeviceMatrix& other) const {
  assert(cols_ == other.rows_);
  assert(depth_ == 1);
  int c_rows = rows_;
  int c_cols = other.cols_;
  DeviceMatrix result(c_rows, c_cols, 1);

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

DeviceMatrix DeviceMatrix::Map(::matrix_mappers::MapperFunc map) const {
  DeviceMatrix result(rows_, cols_, depth_);
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

float DeviceMatrix::Sum() const {
  DeviceMatrix result(1, 1, 1);
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

DeviceMatrix DeviceMatrix::Sum(int layers) const {
  assert(layers >= 0);
  if (layers == 0) {
    // sum columns
    assert(depth_ == 1);
    DeviceMatrix result(rows_, 1, 1);
    MatrixSumColumns<<<(rows_ + 255) / 256, 256>>>(
        data_.get(),
        rows_, cols_,
        result.data_.get());
    return result;
  } else {
    assert(depth_ % layers == 0);
    DeviceMatrix result(1, 1, layers);
    MatrixSumLayers<<<(layers + 255) / 256, 256>>>(
        data_.get(),
        rows_, cols_, depth_,
        result.data_.get(),
        layers);
    return result;
  }
}

__global__ void VecL2(float* A, int len, float* B) {
  float result = 0.0;
  for (int i = 0; i < len; ++i) {
    result += A[i] * A[i];
  }
  B[0] = sqrt(result);
}

float DeviceMatrix::L2() const {
  DeviceMatrix result(1, 1, 1);
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

float DeviceMatrix::Softmax(const DeviceMatrix& expected_class) const {
  assert(depth_ == 1);
  // rows_ = number of classes
  // cols_ = number of samples (we run the same algorithm for each sample)
  assert(expected_class.rows_ == 1);
  assert(expected_class.cols_ == cols_);
  assert(expected_class.depth_ == 1);

  DeviceMatrix result(1, cols_, 1);
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

DeviceMatrix DeviceMatrix::SoftmaxGradient(const DeviceMatrix& expected_class) const {
  // Covered in cnn/error_layer_test.cc.

  assert(depth_ == 1);
  // rows_ = number of classes
  // cols_ = number of samples (we run the same algorithm for each sample)
  assert(expected_class.rows_ == 1);
  assert(expected_class.cols_ == cols_);
  assert(expected_class.depth_ == 1);

  DeviceMatrix result(rows_, cols_, 1);
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

float DeviceMatrix::NumMatches(const DeviceMatrix& expected_class) const {
  assert(depth_ == 1);
  // rows_ = number of classes
  // cols_ = number of samples (we run the same algorithm for each sample)
  assert(expected_class.rows_ == 1);
  assert(expected_class.cols_ == cols_);
  assert(expected_class.depth_ == 1);

  DeviceMatrix result(1, cols_, 1);
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

void DeviceMatrix::Fill(float value) {
  VecFill<<<(size_ + 255) / 256, 256>>>(value, data_.get(), size_);
}

__global__ void VecFillColumn(float value, int col, float* A, int rows, int cols, int depth) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < rows) {
    for (int j = 0; j < depth; ++j) { // FIXME
      A[Dim3toDim1(i, col, j, rows, cols, depth)] = value;
    }
  }
}

void DeviceMatrix::FillColumn(int col, float value) {
  assert(col >= 0 && col < cols_);
  VecFillColumn<<<(rows_ + 255) / 256, 256>>>(value, col, data_.get(), rows_, cols_, depth_);
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

DeviceMatrix DeviceMatrix::AddPadding(
    int row_padding, int col_padding) const {
  if (row_padding <= 0 && col_padding <= 0) {
    return *this;
  }

  DeviceMatrix result(
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

__global__ void MatrixConstRow(
    float* A,
    int rows, int cols, int depth,
    float value,
    float* B) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int k = threadIdx.z + blockDim.z * blockIdx.z;

  if (i < rows + 1 && j < cols && k < depth) {
    int b_index = Dim3toDim1(i, j, k, rows + 1, cols, depth);
    if (i < rows) {
      int a_index = Dim3toDim1(i, j, k, rows, cols, depth);
      B[b_index] = A[a_index];
    } else {
      B[b_index] = value;
    }
  }
}

DeviceMatrix DeviceMatrix::AddConstRow(float value) const {
  DeviceMatrix result(
      rows_ + 1,
      cols_,
      depth_);  // filled with zeros

  dim3 threadsPerBlock(16, 16, 1);
  dim3 blocks((rows_ + 1 + 15) / 16, (cols_ + 15) / 16, depth_);
  MatrixConstRow<<<blocks, threadsPerBlock>>>(
      data_.get(),
      rows_, cols_, depth_,
      value,
      result.data_.get());
  return result;
}

__global__ void MatrixReduceSize(
    float* A,
    int a_rows, int a_cols, int a_depth,
    float* B,
    int b_rows, int b_cols, int b_depth) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int k = threadIdx.z + blockDim.z * blockIdx.z;

  if (i < b_rows && j < b_cols && k < b_depth) {
    int a_index = Dim3toDim1(i, j, k, a_rows, a_cols, a_depth);
    int b_index = Dim3toDim1(i, j, k, b_rows, b_cols, b_depth);
    B[b_index] = A[a_index];
  }
}

DeviceMatrix DeviceMatrix::ReduceSize(int rows, int cols, int depth) const {
  assert(rows <= rows_);
  assert(cols <= cols_);
  assert(depth <= depth_);
  DeviceMatrix result(rows, cols, depth);

  dim3 threadsPerBlock(16, 16, 1);
  dim3 blocks((rows + 15) / 16, (cols + 15) / 16, depth);
  MatrixReduceSize<<<blocks, threadsPerBlock>>>(
      data_.get(),
      rows_, cols_, depth_,
      result.data_.get(),
      rows, cols, depth);
  return result;
}

__global__ void MatrixConvolution(
    int layers_per_image,
    float* A, int a_rows, int a_cols, int a_depth,
    float* filters, int f_rows, int f_cols, int f_depth,
    float* B, int b_rows, int b_cols, int b_depth,
    float* biases) {
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
    if (biases != NULL) {
      sum += biases[filter_id];
    }
    B[Dim3toDim1(i, j, k, b_rows, b_cols, b_depth)] = sum;
  }
}

DeviceMatrix DeviceMatrix::Convolution(
    const DeviceMatrix& filters,
    int layers_per_image,
    int stride,
    const DeviceMatrix& biases) const {
  int row_slots = rows_ - filters.rows() + 1;
  int col_slots = cols_ - filters.cols() + 1;
  assert(row_slots % stride == 0 && col_slots % stride == 0);

  assert(filters.depth() % layers_per_image == 0);
  assert(depth() % layers_per_image == 0);
  float* biases_ptr = NULL;
  if (biases.depth() > 0) {
    assert(biases.depth() * layers_per_image == filters.depth());
    biases_ptr = biases.data_.get();
  }

  assert(stride == 1);  // TODO
  DeviceMatrix result(
      row_slots / stride,
      col_slots / stride,
      filters.depth() / layers_per_image * depth() / layers_per_image);
  dim3 threadsPerBlock(16, 16, 1);
  dim3 blocks((result.rows() + 15) / 16, (result.cols() + 15) / 16, result.depth());
  MatrixConvolution<<<blocks, threadsPerBlock>>>(
      layers_per_image,
      data_.get(), rows_, cols_, depth_,
      filters.data_.get(), filters.rows(), filters.cols(), filters.depth(),
      result.data_.get(), result.rows(), result.cols(), result.depth(),
      biases_ptr);

  return result;
}

DeviceMatrix DeviceMatrix::Convolution(
    const DeviceMatrix& filters,
    int layers_per_image,
    int stride) const {
  return Convolution(filters, layers_per_image, stride, DeviceMatrix());
}

DeviceMatrix DeviceMatrix::ReshapeToColumns(int unit_depth) const {
  assert(depth_ % unit_depth == 0);
  DeviceMatrix rows(*this);
  rows.cols_ = rows_ * cols_ * unit_depth;
  rows.rows_ = depth_ / unit_depth;
  rows.depth_ = 1;
  return rows.T();
}

DeviceMatrix DeviceMatrix::ReshapeFromColumns(int unit_rows, int unit_cols, int unit_depth) const {

  assert(unit_rows * unit_cols * unit_depth == rows_);

  DeviceMatrix rows(this->T());
  rows.depth_ = rows.rows_ * rows.cols_ / (unit_rows * unit_cols);
  rows.rows_ = unit_rows;
  rows.cols_ = unit_cols;
  return rows;
}

DeviceMatrix DeviceMatrix::ReorderLayers(int layers_per_image) const {
  assert(depth_ % layers_per_image == 0);
  DeviceMatrix result(rows_, cols_, depth_);
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

DeviceMatrix DeviceMatrix::DeepCopy() const {
  DeviceMatrix result(rows_, cols_, depth_);
  cudaMemcpy(
      result.data_.get(),
      data_.get(),
      size_ * sizeof(float),
      cudaMemcpyDeviceToDevice);
  return result;
}

float DeviceMatrix::GetValue(int row, int col, int depth) const {
  float result;
  cudaMemcpy(
      &result,
      data_.get() + Index(row, col, depth),
      sizeof(float),
      cudaMemcpyDeviceToHost);
  return result;
}

void DeviceMatrix::SetValue(int row, int col, int depth, float value) {
  cudaMemcpy(
      data_.get() + Index(row, col, depth),
      &value,
      sizeof(float),
      cudaMemcpyHostToDevice);
}

