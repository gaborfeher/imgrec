#ifndef _CNN_CONVOLUTIONAL_LAYER_H_
#define _CNN_CONVOLUTIONAL_LAYER_H_

#include "gtest/gtest_prod.h"

#include "cnn/layer.h"
#include "cnn/matrix_param.h"

class Random;

class ConvolutionalLayer : public Layer {
 public:
  ConvolutionalLayer(
      int num_filters,
      int filter_width,
      int filter_height,
      int padding,
      int layers_per_image);
  virtual void Print() const;
  virtual void Initialize(Random* random);
  virtual void Forward(const Matrix& input);
  virtual void Backward(const Matrix& output_gradient);
  virtual void ApplyGradient(const GradientInfo& info);
  virtual int NumParameters() const;

 private:
  FRIEND_TEST(ConvolutionalLayerTest, IntegratedGradientTest);
  friend class ConvolutionalLayerGradientTest;

  int padding_;
  int layers_per_image_;
  int stride_;
  MatrixParam filters_;
};

#endif  // _CNN_CONVOLUTIONAL_LAYER_H_
