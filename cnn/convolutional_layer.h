#ifndef _CNN_CONVOLUTIONAL_LAYER_H_
#define _CNN_CONVOLUTIONAL_LAYER_H_

#include "gtest/gtest_prod.h"

#include "cnn/layer.h"
#include "linalg/device_matrix.h"

class ConvolutionalLayer : public Layer {
 public:
  ConvolutionalLayer(
      int num_filters,
      int filter_width,
      int filter_height,
      int padding,
      int layers_per_image,
      int stride);
  virtual void Forward(const DeviceMatrix& input);
  virtual void Backward(const DeviceMatrix& output_gradients);
  virtual void ApplyGradient(float learn_rate);

 private:
  FRIEND_TEST(ConvolutionalLayerTest, IntegratedGradientTest);
  FRIEND_TEST(ConvolutionalLayerTest, TrainTest);
  friend class ConvolutionalLayerGradientTest;

  int padding_;
  int layers_per_image_;
  int stride_;
  DeviceMatrix filters_;
  DeviceMatrix filters_gradients_;
};

#endif  // _CNN_CONVOLUTIONAL_LAYER_H_
