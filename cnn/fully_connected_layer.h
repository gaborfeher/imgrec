#ifndef _CNN_FULLY_CONNECTED_LAYER_H_
#define _CNN_FULLY_CONNECTED_LAYER_H_

#include "cnn/layer.h"
#include "gtest/gtest_prod.h"
#include "linalg/device_matrix.h"

class Random;

class FullyConnectedLayer : public Layer {
 public:
  FullyConnectedLayer(int input_size, int output_size);
  virtual void Initialize(Random* random);
  virtual void Forward(const DeviceMatrix& input);
  virtual void Backward(const DeviceMatrix& output_gradients); 
  virtual void ApplyGradient(float learn_rate);
  virtual void Regularize(float lambda);

 private:
  FRIEND_TEST(LearnTest, FullyConnectedLayerWeightGradient);
  FRIEND_TEST(LearnTest, FullyConnectedLayerInputGradient);
  FRIEND_TEST(ConvolutionalLayerTest, TrainTest);

  bool bias_trick_;
  int input_size_;
  int output_size_;
  DeviceMatrix weights_;
  DeviceMatrix weights_gradients_;
};



#endif  // _CNN_FULLY_CONNECTED_LAYER_H_
