#ifndef _CNN_LAYER_H_
#define _CNN_LAYER_H_

#include "linalg/device_matrix.h"

class Layer {
 public:
  virtual ~Layer() {}

  virtual void forward(const DeviceMatrix& input) = 0;
  virtual void backward(const DeviceMatrix& ouotput_gradients) = 0;

  virtual DeviceMatrix output() { return output_; }
  virtual DeviceMatrix input_gradients() { return input_gradients_; }

 private:
  DeviceMatrix input_;
  DeviceMatrix output_;
  DeviceMatrix input_gradients_;


};




#endif  // _CNN_LAYER_H_
