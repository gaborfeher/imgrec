#ifndef _LINALG_DEVICE_MATRIX_H_
#define _LINALG_DEVICE_MATRIX_H_

#include <memory>
#include <vector>

class DeviceMatrix {
 public:
  DeviceMatrix();  // "NULL" matrix
  DeviceMatrix(int rows, int cols);  // rows x cols un-initialized values
  DeviceMatrix(int rows, int cols, float* data);

  int rows() const { return rows_; }
  int cols() const { return cols_; }

  std::shared_ptr<float> get_host_data() const;
  std::vector<float> GetVector() const;
  void Print() const;

  DeviceMatrix Add(const DeviceMatrix& other) const;
  DeviceMatrix ElementwiseMultiply(const DeviceMatrix& other) const;
  DeviceMatrix Multiply(float) const;
  // Returns L2 norm in 1x1 matrix (for now)
  DeviceMatrix L2() const;
  DeviceMatrix T() const;
  DeviceMatrix Dot(const DeviceMatrix&) const;
  DeviceMatrix ApplySigmoid() const;
  DeviceMatrix ApplySigmoidGradients() const;

  void Fill(float value);

  void AssertDimensions(int rows, int cols) const;
  void AssertSameDimensions(const DeviceMatrix& other) const;
  void AssertRows(int rows) const;

  // Shallow-copy is supported by the compiler-generated
  // copy constructor and assignment operator.

 private:
  int rows_;
  int cols_;
  int size_;
  std::shared_ptr<float> data_;
};

#endif // _LINALG_DEVICE_MATRIX_H_
