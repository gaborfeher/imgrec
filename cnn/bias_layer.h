#ifndef _CNN_BIAS_LAYER_H_
#define _CNN_BIAS_LAYER_H_

#include "cnn/bias_like_layer.h"

#include "gtest/gtest_prod.h"

class Matrix;
class Random;

class BiasLayer : public BiasLikeLayer {
 public:
  // See BiasLikeLayer for param docs.
  BiasLayer(int num_neurons, bool layered);
  virtual void Print() const;
  virtual void Initialize(Random* /* generator */);
  virtual void Forward(const Matrix& input);
  virtual void Backward(const Matrix& ouotput_gradient);
  virtual void ApplyGradient(float learn_rate);
  virtual int NumParameters() const;
 private:
  FRIEND_TEST(BiasLayerTest, GradientCheck_ColumnMode);
  FRIEND_TEST(BiasLayerTest, GradientCheck_LayerMode);
  FRIEND_TEST(BiasLayerTest, Forwardpass_ColumnMode);
  FRIEND_TEST(BiasLayerTest, Forwardpass_LayerMode);
  FRIEND_TEST(ConvolutionalLayerTest, IntegratedGradientTest);

  Matrix biases_;
  Matrix biases_gradient_;
};

#endif  // _CNN_BIAS_LAYER_H_
