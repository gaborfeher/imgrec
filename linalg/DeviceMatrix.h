#ifndef _DEVICE_MATRIX_H_
#define _DEVICE_MATRIX_H_

class HostMatrix;

class DeviceMatrix {
public:
  DeviceMatrix(int rows, int cols, float* data);
  explicit DeviceMatrix(const HostMatrix& src);

  ~DeviceMatrix();

  // Disable copy and move.
  DeviceMatrix(const DeviceMatrix&) = delete;
  DeviceMatrix& operator=(const DeviceMatrix&) = delete;
public:  // TODO
  int rows_;
  int cols_;
  int size_;
  float* data_;
};

#endif // _DEVICE_MATRIX_H_
