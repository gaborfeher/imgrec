#ifndef _CNN_FULLY_CONNECTED_LAYER_H_
#define _CNN_FULLY_CONNECTED_LAYER_H_

#include "cnn/layer.h"
#include "cnn/matrix_param.h"

#include "gtest/gtest_prod.h"

class Random;
class Matrix;
struct GradientInfo;
namespace cereal {
class PortableBinaryOutputArchive;
class PortableBinaryInputArchive;
class access;
}

class FullyConnectedLayer : public Layer {
 public:
  FullyConnectedLayer(int input_size, int output_size);
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
  FullyConnectedLayer() {}  // for cereal
  friend class cereal::access;
  FRIEND_TEST(FullyConnectedLayerTest, InputGradient);
  FRIEND_TEST(FullyConnectedLayerTest, WeightGradient);
  FRIEND_TEST(LayerStackTest, SaveLoad);

  int input_size_;
  int output_size_;
  MatrixParam weights_;
};

#endif  // _CNN_FULLY_CONNECTED_LAYER_H_
