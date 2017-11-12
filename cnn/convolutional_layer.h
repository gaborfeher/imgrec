#ifndef _CNN_CONVOLUTIONAL_LAYER_H_
#define _CNN_CONVOLUTIONAL_LAYER_H_

#include "gtest/gtest_prod.h"

#include "cnn/layer.h"
#include "cnn/matrix_param.h"

class Random;
namespace cereal {
class PortableBinaryOutputArchive;
class PortableBinaryInputArchive;
class access;
}

class ConvolutionalLayer : public Layer {
 public:
  ConvolutionalLayer(
      int num_filters,
      int filter_width,
      int filter_height,
      int padding,
      int layers_per_image);
  virtual std::string Name() const;
  virtual void Print() const;
  virtual void Initialize(std::shared_ptr<Random> random);
  virtual void Forward(const Matrix& input);
  virtual void Backward(const Matrix& output_gradient);
  virtual void ApplyGradient(const GradientInfo& info);
  virtual int NumParameters() const;

  // serialization/deserialization
  void save(cereal::PortableBinaryOutputArchive& ar) const;
  void load(cereal::PortableBinaryInputArchive& ar);

 private:
  ConvolutionalLayer() {}  // for cereal
  friend class cereal::access;
  FRIEND_TEST(ConvolutionalLayerTest, IntegratedGradientTest);
  friend class ConvolutionalLayerGradientTest;
  FRIEND_TEST(LayerStackTest, SaveLoad);

  int padding_;
  int layers_per_image_;
  MatrixParam filters_;
};

#endif  // _CNN_CONVOLUTIONAL_LAYER_H_
