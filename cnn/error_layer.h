#ifndef _CNN_ERROR_LAYER_H_
#define _CNN_ERROR_LAYER_H_

#include "cnn/layer.h"

class ErrorLayer : public Layer {
 public:
  ErrorLayer();
  virtual float GetError() const;
  virtual float GetAccuracy() const = 0;
  virtual void SetExpectedValue(const DeviceMatrix& expected_value) = 0;

 protected:
  float error_;
};

#endif  // _CNN_ERROR_LAYER_H_
