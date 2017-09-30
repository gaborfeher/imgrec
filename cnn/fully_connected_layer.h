#ifndef _CNN_FULLY_CONNECTED_LAYER_H_
#define _CNN_FULLY_CONNECTED_LAYER_H_

#include "cnn/layer.h"
#include "linalg/device_matrix.h"

class FullyConnectedLayer : public Layer {
 public:
  FullyConnectedLayer(int input_rows, int input_cols, int output_rows, int output_cols);
  virtual void Forward(const DeviceMatrix& input);
  virtual void Backward(const DeviceMatrix& output_gradients); 
  virtual void ApplyGradient(float learn_rate);

 private:
  DeviceMatrix weights_;
  DeviceMatrix weights_gradients_;
  
};



#endif  // _CNN_FULLY_CONNECTED_LAYER_H_
