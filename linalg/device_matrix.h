#ifndef _LINALG_DEVICE_MATRIX_H_
#define _LINALG_DEVICE_MATRIX_H_

#include <memory>

#include "linalg/base_matrix.h"

class HostMatrix;

class DeviceMatrix : public BaseMatrix {
public:
  DeviceMatrix(int rows, int cols);
  explicit DeviceMatrix(const HostMatrix& src);

  DeviceMatrix Add(const DeviceMatrix& other) const;

  DeviceMatrix Multiply(float) const {
    // TODO
    return DeviceMatrix(1, 1);
  }

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
