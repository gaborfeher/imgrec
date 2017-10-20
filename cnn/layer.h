#ifndef _CNN_LAYER_H_
#define _CNN_LAYER_H_

#include "linalg/device_matrix.h"

class Random;

class Layer {
 public:
  enum Phase {
    NONE,
    PRE_TRAIN_PHASE,
    TRAIN_PHASE,
    POST_TRAIN_PHASE,
    INFER_PHASE,
  };

  Layer();
  virtual ~Layer() {}

  virtual void Print() const {};
  virtual void Initialize(Random* /* generator */) {};
  virtual void Forward(const DeviceMatrix& input) = 0;
  virtual void Backward(const DeviceMatrix& ouotput_gradient) = 0;
  virtual void ApplyGradient(float /* learn_rate */) {};
  virtual void Regularize(float /* lambda */) {};


  // Signals to the layer that a phase is beginning.
  // For PRE_TRAIN_PHASE and POST_TRAIN_PHASE, the return value
  // determines the forward passes of the phase: it will be the
  // highest value returned by a layer (can be 0).
  // Even layers returning =0 must be able to handle all the
  // phases.
  virtual int BeginPhase(Phase /* phase */) {
    return 0;
  };
  virtual void EndPhase(Phase /* phase */) {};

  virtual DeviceMatrix output() { return output_; }
  virtual DeviceMatrix input_gradients() { return input_gradients_; }

 protected:
  DeviceMatrix input_;
  DeviceMatrix output_;
  DeviceMatrix input_gradients_;


};




#endif  // _CNN_LAYER_H_
