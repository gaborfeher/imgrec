#ifndef _LINALG_DEVICE_MATRIX_H_
#define _LINALG_DEVICE_MATRIX_H_

#include <memory>
#include <random>
#include <vector>

#include "util/random.h"

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

class DeviceMatrix {
 public:
  DeviceMatrix();  // "NULL" matrix
  DeviceMatrix(int rows, int cols, int depth);  // rows x cols x depth un-initialized values
  DeviceMatrix(int rows, int cols, int depth, float* data);
  DeviceMatrix(int rows, int cols, int depth, const std::vector<float>& data);

  int rows() const { return rows_; }
  int cols() const { return cols_; }
  int depth() const { return depth_; }

  std::shared_ptr<float> get_host_data() const;
  std::vector<float> GetVector() const;
  void SetVector(const std::vector<float>& data);
  void Print() const;

  DeviceMatrix Add(const DeviceMatrix& other) const;
  DeviceMatrix AddConst(float c) const;
  DeviceMatrix Pow(float exp) const;
  DeviceMatrix ElementwiseMultiply(const DeviceMatrix& other) const;
  DeviceMatrix ElementwiseDivide(const DeviceMatrix& other) const;
  DeviceMatrix Multiply(float) const;
  float Softmax(const DeviceMatrix& expected_class) const;
  DeviceMatrix SoftmaxGradient(const DeviceMatrix& expected_class) const;
  float NumMatches(const DeviceMatrix& expected_class) const;

  float L2() const;
  float Sum() const;
  // layers > 0: consider this matrix as a series of matrices
  //    of depth |layers|, and summarize them into one matrix
  //    of depth layer. After that, calculate the sum of each layer.
  // layers = 0: summarize the columns of the matrix, the
  //             resulting matrix will have one column. This
  //             matrix must have depth = 1.
  DeviceMatrix Sum(int layers) const;
  // layers > 0: repeat this matrix as a series of layers
  //    in the resulting matrix, times times
  // layers = 0: repeat this matrix as a series of columns
  //    in the resulting matrix. This matrix must have depth = 1.
  DeviceMatrix Repeat(int times, int layers) const;

  DeviceMatrix T() const;
  DeviceMatrix Rot180() const;
  DeviceMatrix Dot(const DeviceMatrix&) const;
  DeviceMatrix Map(::matrix_mappers::MapperFunc map) const;
  DeviceMatrix AddPadding(int row_padding, int col_padding) const;
  // Adds a constant row att the bottom of each layer.
  DeviceMatrix AddConstRow(float value) const;
  // It can drop last rows, last cols or last layers.
  DeviceMatrix ReduceSize(int rows, int cols, int depth) const;
  DeviceMatrix ReshapeToColumns(int unit_depth) const;
  DeviceMatrix ReshapeFromColumns(int unit_rows, int unit_cols, int unit_depth) const;
  // Assuming that the matrix has n images, k layers each,
  // roerders the layers to have k images with n layers each.
  // The first new image will consist of the first layers of each
  // original image (in the same order), the second img will
  // consist of the second image of each layer, etc.
  // TODO: integrate this into Convolution to get rid of copies
  DeviceMatrix ReorderLayers(int layers_per_image) const;

  // depth of filters must be a multiple of depth of this matrix,
  // and it contains that many filters.
  DeviceMatrix Convolution(
      const DeviceMatrix& filters,
      int layers_per_image,
      int stride) const;
  DeviceMatrix Convolution(
      const DeviceMatrix& filters,
      int layers_per_image,
      int stride,
      const DeviceMatrix& biases) const;

  void Fill(float value);
  void FillColumn(int col, float value);

  template <class Distribution>
  void RandomFill(Random* random, const Distribution& distribution) {
    std::vector<float> result;
    result.reserve(size_);
    for (int i = 0; i < size_; ++i) {
      result.push_back(random->RandFloat(distribution));
    }
    SetVector(result);
  }

  DeviceMatrix DeepCopy() const;

  float GetValue(int row, int col, int depth) const;
  void SetValue(int row, int col, int depth, float value);

  void AssertDimensions(int rows, int cols, int depth) const;
  void AssertSameDimensions(const DeviceMatrix& other) const;
  void AssertRows(int rows) const;
  void AssertDepth(int depth) const;

  // Shallow-copy is supported by the compiler-generated
  // copy constructor and assignment operator.

 private:
  int Index(int i, int j, int k) const;

  int rows_;
  int cols_;
  int depth_;
  int size_;
  std::shared_ptr<float> data_;
};

#endif // _LINALG_DEVICE_MATRIX_H_
