#ifndef _LINALG_MATRIX_H_
#define _LINALG_MATRIX_H_

#include <memory>
#include <random>
#include <vector>

#include "util/random.h"

#include "gtest/gtest_prod.h"

namespace matrix_mappers {

typedef void (*MapperFunc)(float* A, float* B, int size);

MapperFunc Sigmoid();
MapperFunc SigmoidGradient();
MapperFunc ReLU();
MapperFunc ReLUGradient();
MapperFunc LReLU();
MapperFunc LReLUGradient();
MapperFunc Square();
MapperFunc Sqrt();

}  // namespace matrix_mappers

class Matrix {
 public:
  Matrix();  // "NULL" matrix
  Matrix(int rows, int cols, int depth);  // rows x cols x depth un-initialized values
  Matrix(int rows, int cols, int depth, const std::vector<float>& data);

  int rows() const { return rows_; }
  int cols() const { return cols_; }
  int depth() const { return depth_; }
  int size() const { return size_; }

  void Print() const;

  Matrix Add(const Matrix& other) const;
  Matrix AddConst(float c) const;
  Matrix Pow(float exp) const;
  Matrix ElementwiseMultiply(const Matrix& other) const;
  Matrix ElementwiseDivide(const Matrix& other) const;
  Matrix Multiply(float) const;
  Matrix Divide(float) const;
  float Softmax(const Matrix& expected_class) const;
  Matrix SoftmaxGradient(const Matrix& expected_class) const;
  float NumMatches(const Matrix& expected_class) const;

  float L2() const;  // L2^2, actually
  float Sum() const;
  // layered = true, layers > 0: consider this matrix as a
  //    series of matrices of depth |layers|, and summarize them
  //    into one matrix of depth layer. After that, calculate the
  //    sum of each layer, so the result becomes 1x1x|layers|.
  // layered = false, layers > 0: summarize the columns
  //    of the matrix, the resulting matrix will have one column.
  //    This matrix must have depth = 1 and rows = layers. (The
  //    value of layers is ignored other than the rows=layers
  //    assertion.)
  Matrix Sum(bool layered, int layers) const;

  // If layered = true, depth = depth_ * k, rows_ = cols_ = 1:
  //    The result will be a rows x cols x depth
  //    matrix, each layer will be filled with the corresponding
  //    element from this, in a pattern repeating in cycles of
  //    depth.
  // If layered = false, cols = cols_ * k, depth = depth_ = cols_ = 1:
  //    The result will be a rows x cols x 1 matrix, each column
  //    is a copy of the original matrix.
  Matrix Repeat(bool layered, int rows, int cols, int depth) const;

  // Consider this matrix as a series of matrices of depth |layers|
  // and summarize them into one matrix of depth |layer|.
  Matrix PerLayerSum(int layers) const;
  // Take this matrix and repeat it |times| times, and put the
  // repeated instances into the result along the depth axis.
  Matrix PerLayerRepeat(int times) const;

  Matrix T() const;
  Matrix Rot180() const;
  Matrix Dot(const Matrix&) const;
  Matrix Map(::matrix_mappers::MapperFunc map) const;
  Matrix AddPadding(int row_padding, int col_padding) const;
  Matrix RemovePadding(int row_padding, int col_padding) const;
  Matrix ReshapeToColumns(int unit_depth) const;
  Matrix ReshapeFromColumns(int unit_rows, int unit_cols, int unit_depth) const;

  // Computes convolution of a and b matrices. (Actually, this is not officially
  // convolution, just cross-corellation. The difference is that b is not rotated
  // by 180degs.)
  // If a and/or b have multiple layers, the following logic applies: the number
  // of layers in both matrices must be a multiple of layers_per_image. a contains
  // the input images, b contains the filters. The convolution of an image and a filter
  // is the sum of their layers' pairwise convolutions. The output will contain the
  // convolution of each input image with each filter, sorted by images first.
  // a_major and b_major controls the order in which the layers of and b are interpreted:
  // major = true: sorted by images first, layers second
  // major = false: sorted by layers first, images second
  // (The output is always major-ordered.)
  static Matrix Convolution(
      const Matrix& a, bool a_major,
      const Matrix& b, bool b_major,
      int layers_per_image);

  // On each layer, independently: slices the matrix into
  // pool_rows x pool_cols sub-matrices and takes the max value
  // from them into the first result. The second result matrix
  // will contain the indices of the max values. (col + row + pool_cols).
  std::pair<Matrix, Matrix> Pooling(int pool_rows, int pool_cols) const;
  // Creates a bigger matrix of size
  // (rows*pool_rows, cols*pool_cols, depth). The values of this
  // matrix are copied into the result according to the positions
  // defined in switches. (The places of max values if switches was
  // the second result of Pooling.) switches should have the same
  // dimensions as this matrix.
  Matrix PoolingSwitch(const Matrix& switches, int pool_rows, int pool_cols) const;

  Matrix DeepCopy() const;

  // Mutating operations:
  void Fill(float value);

  template <class Distribution>
  void RandomFill(Random* random, const Distribution& distribution) {
    std::vector<float> result;
    result.reserve(size_);
    for (int i = 0; i < size_; ++i) {
      result.push_back(random->RandFloat(distribution));
    }
    SetVector(result);
  }

  // Each element is populated with 0 or 1/p. Probability
  // of zero is 1-p, probability of 1/p is p.
  void InvertedDropoutFill(Random* random, float p);

  float GetValue(int row, int col, int depth) const;
  void SetValue(int row, int col, int depth, float value);

  void AssertDimensions(int rows, int cols, int depth) const;
  void AssertSameDimensions(const Matrix& other) const;
  void AssertRows(int rows) const;
  void AssertDepth(int depth) const;

  // Shallow-copy is supported by the compiler-generated
  // copy constructor and assignment operator.

 private:
  FRIEND_TEST(SmallMatrixTest, HostDeviceTransfer);
  FRIEND_TEST(SmallMatrixTest, InvertedDropoutFill);
  // TODO: use public API for these tests:
  friend void ExpectMatrixEquals(const Matrix&, const Matrix&);
  friend void ExpectMatrixEquals(const Matrix&, const Matrix&, float, float);
  // TODO: without friendship:
  friend struct MatrixPack;

  int Index(int i, int j, int k) const;
  std::shared_ptr<float> get_host_data() const;
  std::vector<float> GetVector() const;
  void SetVector(const std::vector<float>& data);

  int rows_;
  int cols_;
  int depth_;
  int size_;
  std::shared_ptr<float> data_;
};

#endif // _LINALG_MATRIX_H_
