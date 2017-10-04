#ifndef _LINALG_DEVICE_MATRIX_H_
#define _LINALG_DEVICE_MATRIX_H_

#include <memory>
#include <vector>

class DeviceMatrix {
 public:
  DeviceMatrix();  // "NULL" matrix
  DeviceMatrix(int rows, int cols);  // rows x cols un-initialized values
  DeviceMatrix(int rows, int cols, int depth);
  DeviceMatrix(int rows, int cols, float* data);
  DeviceMatrix(int rows, int cols, int depth, float* data);

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
  DeviceMatrix ApplySigmoid() const;
  DeviceMatrix ApplySigmoidGradients() const;
  DeviceMatrix AddPadding(int padding) const;
  DeviceMatrix ReshapeToColumns(int unit_depth) const;
  DeviceMatrix ReshapeFromColumns(int unit_rows, int unit_cols, int unit_depth) const;
  // Assumas that there are images of depth unit_depth in blocks of
  // size units_per_block. Reorders the images so that it first
  // takes all the first images of each block, then second images
  // of each block, etc.
  DeviceMatrix ReorderLayers(int unit_depth, int units_per_block) const;

  // depth of filters must be a multiple of depht of this matrix,
  // and it contains that many filters.
  DeviceMatrix Convolution(
      const DeviceMatrix& filters,
      int layers_per_image,
      int stride) const;

  void Fill(float value);

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
