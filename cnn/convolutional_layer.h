#ifndef _CNN_CONVOLUTIONAL_LAYER_H_
#define _CNN_CONVOLUTIONAL_LAYER_H_

#include "gtest/gtest_prod.h"

#include "cnn/layer.h"
#include "linalg/device_matrix.h"

class Random;

class ConvolutionalLayer : public Layer {
 public:
  ConvolutionalLayer(
      int num_filters,
      int filter_width,
      int filter_height,
      int padding,
      int layers_per_image,
      int stride);
  virtual void Print() const;
  virtual void Initialize(Random* random);
  virtual void Forward(const DeviceMatrix& input);
  virtual void Backward(const DeviceMatrix& output_gradient);
  virtual void ApplyGradient(float learn_rate);
  virtual void Regularize(float lambda);

 private:
  FRIEND_TEST(ConvolutionalLayerTest, IntegratedGradientTest);
  friend class ConvolutionalLayerGradientTest;

  int padding_;
  int layers_per_image_;
  int stride_;
  DeviceMatrix filters_;
  DeviceMatrix filters_gradient_;
  DeviceMatrix biases_;
  DeviceMatrix biases_gradient_;
};

#endif  // _CNN_CONVOLUTIONAL_LAYER_H_
