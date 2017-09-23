#ifndef _BASE_MATRIX_H_
#define _BASE_MATRIX_H_

class BaseMatrix {
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

#endif  // _BASE_MATRIX_H_
