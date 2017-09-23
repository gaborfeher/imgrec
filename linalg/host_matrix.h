#ifndef _HOST_MATRIX_H_
#define _HOST_MATRIX_H_

#include <memory>
#include <vector>

#include "base_matrix.h"

class DeviceMatrix;

class HostMatrix : public BaseMatrix {
public:
  // Makes a copy of data:
  HostMatrix(int rows, int cols, float* data);
  explicit HostMatrix(const DeviceMatrix& src);
  void Print() const;
  std::vector<float> GetVector() const;

  // Shallow-copy is supported by the compiler-generated
  // copy constructor and assignment operator.
};

#endif // _HOST_MATRIX_H_
