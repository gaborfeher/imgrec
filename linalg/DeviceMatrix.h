#ifndef _DEVICE_MATRIX_H_
#define _DEVICE_MATRIX_H_

#include <memory>

class HostMatrix;

class DeviceMatrix {
public:
  DeviceMatrix(int rows, int cols, float* data);
  explicit DeviceMatrix(const HostMatrix& src);

  // Shallow-copy is supported by the compiler-generated
  // copy constructor and assignment operator.
public:  // TODO
  int rows_;
  int cols_;
  int size_;
  std::shared_ptr<float> data_;
};

#endif // _DEVICE_MATRIX_H_
