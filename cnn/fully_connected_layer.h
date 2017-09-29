#ifndef _CNN_FULLY_CONNECTED_LAYER_H_
#define _CNN_FULLY_CONNECTED_LAYER_H_

#include "cnn/layer.h"
#include "linalg/device_matrix.h"

class FullyConnectedLayer : public Layer {
 public:
  FullyConnectedLayer(int input_size, int output_size);
  virtual void forward(const DeviceMatrix& input);
  virtual void backward(const DeviceMatrix& output_gradients); 
 private:
  int input_size_,
  int output_size_,
  DeviceMatrix weights_;
  DeviceMatrix weights_gradients_;
  
};



#endif  // _CNN_FULLY_CONNECTED_LAYER_H_
