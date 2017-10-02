#ifndef _CNN_SIGMOID_LAYER_H_
#define _CNN_SIGMOID_LAYER_H_

#include "cnn/layer.h"
#include "linalg/device_matrix.h"

class SigmoidLayer : public Layer {
 public:
  SigmoidLayer();
  virtual void Forward(const DeviceMatrix& input);
  virtual void Backward(const DeviceMatrix& output_gradients);
  virtual void ApplyGradient(float learn_rate);
};


#endif  // _CNN_SIGMOID_LAYER_H_
