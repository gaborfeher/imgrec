#ifndef _CNN_INVERTED_DROPOUT_LAYER_H_
#define _CNN_INVERTED_DROPOUT_LAYER_H_

#include "cnn/bias_like_layer.h"

#include <memory>

class Random;

// https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
class InvertedDropoutLayer : public BiasLikeLayer {
 public:
  InvertedDropoutLayer(
      int num_neurons,
      bool layered,
      float p,
      std::shared_ptr<Random> random);
  virtual void Forward(const Matrix& input);
  virtual void Backward(const Matrix& output_gradient);
 private:
  float p_;
  std::shared_ptr<Random> random_;
  Matrix mask_;
};

#endif  // _CNN_INVERTED_DROPOUT_LAYER_H_
