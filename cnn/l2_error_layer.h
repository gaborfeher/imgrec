#ifndef _CNN_L2_ERROR_LAYER_H_
#define _CNN_L2_ERROR_LAYER_H_

#include "cnn/error_layer.h"
#include "linalg/device_matrix.h"

class L2ErrorLayer : public ErrorLayer {
 public:
  L2ErrorLayer();
  virtual void SetExpectedValue(const DeviceMatrix& expected_value);
  virtual void Forward(const DeviceMatrix& input);
  virtual void Backward(const DeviceMatrix& output_gradients); 

  float GetError() const;

 private:
  DeviceMatrix expected_value_;
};

#endif  // _CNN_L2_ERROR_LAYER_H_
