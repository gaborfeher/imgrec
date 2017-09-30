#ifndef _LINALG_BASE_MATRIX_H_
#define _LINALG_BASE_MATRIX_H_

#include <memory>

class BaseMatrix {
public:
  int rows() const { return rows_; }
  int cols() const { return cols_; }

protected:
  BaseMatrix(int rows, int cols) :
      rows_(rows),
      cols_(cols),
      size_(rows * cols) {
  }

  virtual ~BaseMatrix() {
  }

  int rows_;
  int cols_;
  int size_;
  std::shared_ptr<float> data_;

  // TODO: cleaner way?
  friend class HostMatrix;
  friend class DeviceMatrix;
};

#endif  // _LINALG_BASE_MATRIX_H_
