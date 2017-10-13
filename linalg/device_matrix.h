#ifndef _LINALG_DEVICE_MATRIX_H_
#define _LINALG_DEVICE_MATRIX_H_

#include <memory>
#include <vector>

namespace matrix_mappers {

typedef void (*MapperFunc)(float* A, float* B);

MapperFunc Sigmoid();
MapperFunc SigmoidGradient();
MapperFunc ReLU();
MapperFunc ReLUGradient();
MapperFunc LReLU();
MapperFunc LReLUGradient();

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
  void Print() const;

  DeviceMatrix Add(const DeviceMatrix& other) const;
  DeviceMatrix ElementwiseMultiply(const DeviceMatrix& other) const;
  DeviceMatrix Multiply(float) const;
  // Returns L2 norm in 1x1 matrix (for now)
  DeviceMatrix L2() const;
  DeviceMatrix T() const;
  DeviceMatrix Rot180() const;
  DeviceMatrix Dot(const DeviceMatrix&) const;
  DeviceMatrix Map(::matrix_mappers::MapperFunc map) const;
  DeviceMatrix AddPadding(int row_padding, int col_padding) const;
  DeviceMatrix ReshapeToColumns(int unit_depth) const;
  DeviceMatrix ReshapeFromColumns(int unit_rows, int unit_cols, int unit_depth) const;
  // Assuming that the matrix has n images, k layers each,
  // roerders the layers to have k images with n layers each.
  // The first new image will consist of the first layers of each
  // original image (in the same order), the second img will
  // consist of the second image of each layer, etc.
  // TODO: integrate this into Convolution to get rid of copies
  DeviceMatrix ReorderLayers(int layers_per_image) const;

  // depth of filters must be a multiple of depht of this matrix,
  // and it contains that many filters.
  DeviceMatrix Convolution(
      const DeviceMatrix& filters,
      int layers_per_image,
      int stride) const;

  void Fill(float value);

  DeviceMatrix DeepCopy() const;
  float GetValue(int row, int col, int depth) const;
  void SetValue(int row, int col, int depth, float value);

  void AssertDimensions(int rows, int cols, int depth) const;
  void AssertSameDimensions(const DeviceMatrix& other) const;
  void AssertRows(int rows) const;

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
