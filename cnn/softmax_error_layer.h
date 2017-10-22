#ifndef _CNN_SOFTMAX_ERROR_LAYER_H_
#define _CNN_SOFTMAX_ERROR_LAYER_H_

#include "cnn/error_layer.h"
#include "linalg/device_matrix.h"

class SoftmaxErrorLayer : public ErrorLayer {
 public:
  SoftmaxErrorLayer();
  virtual void SetExpectedValue(const DeviceMatrix& expected_value);
  virtual void Forward(const DeviceMatrix& input);
  virtual void Backward(const DeviceMatrix& output_gradient);
  virtual float GetAccuracy() const;

 private:
  DeviceMatrix expected_value_;
};

#endif  // _CNN_SOFTMAX_ERROR_LAYER_H_
