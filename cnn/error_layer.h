#ifndef _CNN_ERROR_LAYER_H_
#define _CNN_ERROR_LAYER_H_

#include "cnn/layer.h"
#include "linalg/device_matrix.h"

class ErrorLayer : public Layer {
 public:
  ErrorLayer();
  void SetExpectedValue(const DeviceMatrix& expected_value);
  virtual void Forward(const DeviceMatrix& input);
  virtual void Backward(const DeviceMatrix& output_gradients); 
  virtual void ApplyGradient(float learn_rate);

 private:
  DeviceMatrix expected_value_;
};

#endif  // _CNN_ERROR_LAYER_H_
