#ifndef _CNN_INPUT_IMAGE_NORMALIZATION_LAYER_H_
#define _CNN_INPUT_IMAGE_NORMALIZATION_LAYER_H_

#include "cnn/layer.h"

class Matrix;

// Computes the mean of all the inputs in the PRE_TRAIN_PHASE,
// and subtracts it in the forward pass of all subsequent phases.
// The input matrix is assumed to be a series of images of depth
// |num_layers|.
class InputImageNormalizationLayer : public Layer {
 public:
  InputImageNormalizationLayer(int rows, int cols, int depth);
  virtual void Print() const;
  virtual void Forward(const Matrix& input);
  virtual void Backward(const Matrix& ouotput_gradient);
  virtual bool BeginPhase(Phase phase, int phase_sub_id);
  virtual void EndPhase(Phase phase, int phase_sub_id);

 private:
  int num_samples_;
  Matrix mean_;  // mean of all inputs times -1

  Phase phase_;
  int phase_sub_id_;
};

#endif  // _CNN_INPUT_IMAGE_NORMALIZATION_LAYER_H_
