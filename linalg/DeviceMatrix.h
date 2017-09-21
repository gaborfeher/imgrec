#ifndef _DEVICE_MATRIX_H_
#define _DEVICE_MATRIX_H_

#include <memory>

#include "BaseMatrix.h"

class HostMatrix;

class DeviceMatrix : public BaseMatrix {
public:
  DeviceMatrix(int rows, int cols, float* data);
  explicit DeviceMatrix(const HostMatrix& src);

  // Shallow-copy is supported by the compiler-generated
  // copy constructor and assignment operator.
};

#endif // _DEVICE_MATRIX_H_
