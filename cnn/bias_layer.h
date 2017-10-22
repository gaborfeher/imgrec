#ifndef _CNN_BIAS_LAYER_H_
#define _CNN_BIAS_LAYER_H_

#include "cnn/layer.h"

#include "gtest/gtest_prod.h"

class DeviceMatrix;
class Random;

class BiasLayer : public Layer {
 public:
  BiasLayer(int neurons, bool convolutional);
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

  bool convolutional_;
  DeviceMatrix biases_;
  DeviceMatrix biases_gradient_;
};

#endif
