#ifndef _HOST_MATRIX_H_
#define _HOST_MATRIX_H_

#include <memory>

class DeviceMatrix;

class HostMatrix {
public:
  // Makes a copy of data:
  HostMatrix(int rows, int cols, float* data);
  explicit HostMatrix(const DeviceMatrix& src);
  void Print();

  // Disable copy and move.
  HostMatrix(const HostMatrix&) = delete;
  HostMatrix& operator=(const HostMatrix&) = delete;
public:  // TODO
  int rows_;
  int cols_;
  int size_;
  std::unique_ptr<float[]> data_;

};

#endif // _HOST_MATRIX_H_
