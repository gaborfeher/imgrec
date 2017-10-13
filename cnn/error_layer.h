#ifndef _CNN_ERROR_LAYER_H_
#define _CNN_ERROR_LAYER_H_

#include "cnn/layer.h"

class ErrorLayer : public Layer {
 public:
  ErrorLayer();
  virtual void ApplyGradient(float learn_rate);
  virtual float GetError() const = 0;
  virtual void SetExpectedValue(const DeviceMatrix& expected_value) = 0;
};

#endif  // _CNN_ERROR_LAYER_H_
