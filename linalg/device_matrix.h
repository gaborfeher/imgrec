#ifndef _LINALG_DEVICE_MATRIX_H_
#define _LINALG_DEVICE_MATRIX_H_

#include <memory>

#include "linalg/base_matrix.h"

class HostMatrix;

class DeviceMatrix : public BaseMatrix {
public:
  explicit DeviceMatrix(const HostMatrix& src);

  DeviceMatrix Add(const DeviceMatrix& other);

  // Shallow-copy is supported by the compiler-generated
  // copy constructor and assignment operator.

private:
  DeviceMatrix(int rows, int cols);
};

#endif // _LINALG_DEVICE_MATRIX_H_
