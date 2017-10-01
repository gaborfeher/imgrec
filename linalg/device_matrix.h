#ifndef _LINALG_DEVICE_MATRIX_H_
#define _LINALG_DEVICE_MATRIX_H_

#include <memory>
#include <vector>

#include "linalg/base_matrix.h"

class DeviceMatrix : public BaseMatrix {
public:
  DeviceMatrix(int rows, int cols);
  DeviceMatrix(int rows, int cols, float* data);

  std::shared_ptr<float> get_host_data() const;
  std::vector<float> GetVector() const;
  void Print() const;

  DeviceMatrix Add(const DeviceMatrix& other) const;

  DeviceMatrix Multiply(float) const;

  // Returns L2 norm in 1x1 matrix (for now)
  DeviceMatrix L2() const {
    // TODO
    return DeviceMatrix(1, 1);
  }

  DeviceMatrix T() const;

  DeviceMatrix Dot(const DeviceMatrix&) const {
    // TODO
    return DeviceMatrix(1, 1);
  }

  DeviceMatrix ApplySigmoid() const {
    // TODO
    return DeviceMatrix(1, 1);
  }

  DeviceMatrix ApplySigmoidGradients() const {
    // TODO
    return DeviceMatrix(1, 1);
  }
  

  // Shallow-copy is supported by the compiler-generated
  // copy constructor and assignment operator.

};

#endif // _LINALG_DEVICE_MATRIX_H_
