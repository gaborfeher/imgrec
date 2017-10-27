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
  // Assuming that the matrix has n images, k layers each,
  // roerders the layers to have k images with n layers each.
  // The first new image will consist of the first layers of each
  // original image (in the same order), the second img will
  // consist of the second image of each layer, etc.
  // TODO: integrate this into Convolution to get rid of copies
  Matrix ReorderLayers(int layers_per_image) const;

  // depth of filters must be a multiple of depth of this matrix,
  // and it contains that many filters.
  Matrix Convolution(const Matrix& filters, int layers_per_image) const;

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

  Matrix DeepCopy() const;

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
  // TODO: use public API for these tests:
  friend void ExpectMatrixEquals(const Matrix&, const Matrix&);
  friend void ExpectMatrixEquals(const Matrix&, const Matrix&, float, float);

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
