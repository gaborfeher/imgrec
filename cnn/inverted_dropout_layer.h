#ifndef _CNN_INVERTED_DROPOUT_LAYER_H_
#define _CNN_INVERTED_DROPOUT_LAYER_H_

#include "cnn/layer.h"

#include <memory>

class Random;

// https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
class InvertedDropoutLayer : public Layer {
 public:
  InvertedDropoutLayer(float p, std::shared_ptr<Random> random);
  virtual void Forward(const Matrix& input);
  virtual void Backward(const Matrix& output_gradient);

  virtual bool BeginPhase(Phase phase, int phase_sub_id);
  virtual void EndPhase(Phase phase, int phase_sub_id);
 private:
  float p_;
  std::shared_ptr<Random> random_;
  Matrix mask_;
  Phase phase_;
  int phase_sub_id_;
};

#endif  // _CNN_INVERTED_DROPOUT_LAYER_H_
