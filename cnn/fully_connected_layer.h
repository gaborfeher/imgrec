#ifndef _CNN_FULLY_CONNECTED_LAYER_H_
#define _CNN_FULLY_CONNECTED_LAYER_H_

#include "cnn/layer.h"
#include "gtest/gtest_prod.h"
#include "linalg/matrix.h"

class Random;

class FullyConnectedLayer : public Layer {
 public:
  FullyConnectedLayer(int input_size, int output_size);
  virtual void Print() const;
  virtual void Initialize(Random* random);
  virtual void Forward(const Matrix& input);
  virtual void Backward(const Matrix& output_gradient);
  virtual void ApplyGradient(float learn_rate);
  virtual void Regularize(float lambda);
  virtual int NumParameters() const;

 private:
  FRIEND_TEST(FullyConnectedLayerTest, WeightGradient);
  FRIEND_TEST(FullyConnectedLayerTest, InputGradient);

  int input_size_;
  int output_size_;
  Matrix weights_;
  Matrix weights_gradient_;
};



#endif  // _CNN_FULLY_CONNECTED_LAYER_H_
