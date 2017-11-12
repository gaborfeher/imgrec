#ifndef _LINALG_MATRIX_H_
#define _LINALG_MATRIX_H_

#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "util/random.h"

#include "gtest/gtest_prod.h"

namespace cereal {
class PortableBinaryOutputArchive;
class PortableBinaryInputArchive;
}

namespace matrix_cuda_util {

void SynchronizeForPerfLogging();

}

namespace matrix_mappers {
typedef void (*Map1Func)(float* a, float* b, int size);
typedef void (*Map2Func)(float* a, float* b, float *c, int size);
typedef void (*Map1PFunc)(float* a, float* b, int size, float param);

Map1Func Sigmoid();
Map1Func SigmoidGradient();
Map1Func ReLU();
Map1Func ReLUGradient();
Map1Func LReLU();
Map1Func LReLUGradient();
Map1Func Square();
Map1Func Sqrt();

}  // namespace matrix_mappers


// This class represents a rows x cols x depth matrix consisting
// of float values, stored in GPU.
// Most operations are leaving the matrix intact (const) and they
// create a new matrix.
// The copy-constructor and assignment operators are supported
// (compiler generated) and they are performing shallow copies.
// Copies of the same matrix refer to the same GPU memory region,
// which is freed when the last copy of the matrix is destroyed.
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

  Matrix /* Elementwise */ Add(const Matrix& other) const;
  Matrix ElementwiseMultiply(const Matrix& other) const;
  Matrix ElementwiseDivide(const Matrix& other) const;

  Matrix Pow(float exp) const;
  Matrix AddConst(float c) const;
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
  Matrix Repeat(bool layered, const Matrix& size_template) const;

  // Consider this matrix as a series of matrices of depth |layers|
  // and summarize them into one matrix of depth |layer|.
  Matrix PerLayerSum(int layers) const;
  // Take this matrix and repeat it |times| times, and put the
  // repeated instances into the result along the depth axis.
  Matrix PerLayerRepeat(int times) const;

  Matrix T() const;
  Matrix Rot180() const;
  Matrix Dot(const Matrix&) const;

  Matrix Map1(::matrix_mappers::Map1Func map) const;
  Matrix Map2(const Matrix& other, ::matrix_mappers::Map2Func map) const;
  Matrix Map1P(float param, ::matrix_mappers::Map1PFunc map) const;

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
      int layers_per_image,
      const Matrix& a, bool a_major,
      const Matrix& b, bool b_major);

  static Matrix Convolution(
      int layers_per_image,
      const Matrix& a, bool a_major, int a_row_padding, int a_col_padding,
      const Matrix& b, bool b_major, int b_row_padding, int b_col_padding,
      int c_row_padding, int c_col_padding);

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
  void RandomFill(
      std::shared_ptr<Random> random,
      const Distribution& distribution) {
    std::vector<float> result;
    result.reserve(size_);
    for (int i = 0; i < size_; ++i) {
      result.push_back(random->RandFloat(distribution));
    }
    SetVector(result);
  }

  // Each element is populated with 0 or 1/p. Probability
  // of zero is 1-p, probability of 1/p is p.
  static Matrix MakeInvertedDropoutMask(
      bool layered, int num_neurons,
      float p, Random* random);

  float GetValue(int row, int col, int depth) const;
  void SetValue(int row, int col, int depth, float value);

  void AssertEquals(const Matrix& other) const;
  void AssertDimensions(int rows, int cols, int depth) const;
  void AssertSameDimensions(const Matrix& other) const;
  void AssertRows(int rows) const;
  void AssertDepth(int depth) const;

  // serialization/deserialization
  void save(cereal::PortableBinaryOutputArchive& ar) const;
  void load(cereal::PortableBinaryInputArchive& ar);

 private:
  FRIEND_TEST(SmallMatrixTest, HostDeviceTransfer);
  FRIEND_TEST(SmallMatrixTest, MakeInvertedDropoutMask_Layered);
  FRIEND_TEST(SmallMatrixTest, MakeInvertedDropoutMask_Columns);
  FRIEND_TEST(SmallMatrixTest, SaveLoad);
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
