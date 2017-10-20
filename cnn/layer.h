#ifndef _CNN_LAYER_H_
#define _CNN_LAYER_H_

#include "linalg/device_matrix.h"

class Random;

class Layer {
 public:
  enum TrainingPhase {
    PRE_PHASE,
    POST_PHASE
  };

  Layer();
  virtual ~Layer() {}

  virtual void Print() const {};
  virtual void Initialize(Random* /* generator */) {};
  virtual void Forward(const DeviceMatrix& input) = 0;
  virtual void Backward(const DeviceMatrix& ouotput_gradient) = 0;
  virtual void ApplyGradient(float /* learn_rate */) {};
  virtual void Regularize(float /* lambda */) {};

  // This should return true if the layer requires
  // the given phase. If no layers return true,
  // then the phase is skipped.
  virtual bool BeginTrainingPhase(TrainingPhase /* phase */) {
    return false;
  };
  virtual void EndTrainingPhase(TrainingPhase /* phase */) {};

  virtual DeviceMatrix output() { return output_; }
  virtual DeviceMatrix input_gradients() { return input_gradients_; }

 protected:
  DeviceMatrix input_;
  DeviceMatrix output_;
  DeviceMatrix input_gradients_;


};




#endif  // _CNN_LAYER_H_
