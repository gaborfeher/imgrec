#ifndef _CNN_BIAS_LAYER_H_
#define _CNN_BIAS_LAYER_H_

#include "cnn/layer.h"

#include "gtest/gtest_prod.h"

class DeviceMatrix;
class Random;

class BiasLayer : public Layer {
 public:
  // num_neurons: Number of distinct neurons in the previous layer
  // layered: Determines if input is a single-layer or
  //          multi-layered matrix. In the multi-layered case,
  //          each layer is considered a neuron and has a common bias.
  //          The input can have k*neurons layers, which means that
  //          the same neurons were applied on several input samples
  //          in sequence.
  BiasLayer(int num_neurons, bool layered);
  virtual void Print() const;
  virtual void Initialize(Random* /* generator */);
  virtual void Forward(const DeviceMatrix& input);
  virtual void Backward(const DeviceMatrix& ouotput_gradient);
  virtual void ApplyGradient(float learn_rate);
 private:
  FRIEND_TEST(BiasLayerTest, GradientCheck_ColumnMode);
  FRIEND_TEST(BiasLayerTest, GradientCheck_LayerMode);
  FRIEND_TEST(BiasLayerTest, Forwardpass_ColumnMode);
  FRIEND_TEST(BiasLayerTest, Forwardpass_LayerMode);
  FRIEND_TEST(ConvolutionalLayerTest, IntegratedGradientTest);

  bool layered_;
  int num_neurons_;

  DeviceMatrix biases_;
  DeviceMatrix biases_gradient_;
};

#endif
