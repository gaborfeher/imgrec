#ifndef _CNN_SIGMOID_LAYER_H_
#define _CNN_SIGMOID_LAYER_H_

#include "cnn/layer.h"
#include "linalg/device_matrix.h"

class SigmoidLayer : public Layer {
 public:
  SigmoidLayer(int size);
  virtual void forward(const DeviceMatrix& input);
  virtual void backward(const DeviceMatrix& output_gradients);

 private:
  int size_;
};


#endif  // _CNN_SIGMOID_LAYER_H_
