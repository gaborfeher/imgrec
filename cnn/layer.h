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
  // For optional phases like PRE_TRAIN_PHASE and POST_TRAIN_PHASE,
  // the return value determines if the phases are needed.
  // (See Model::RunPhase.)
  // Even layers returning false must be able to handle all the
  // phases.
  virtual bool BeginPhase(Phase /* phase */, int /* phase_sub_id */) {
    return false;
  };
  virtual void EndPhase(Phase /* phase */, int /* phase_sub_id */) {};

  virtual DeviceMatrix output() { return output_; }
  virtual DeviceMatrix input_gradient() { return input_gradient_; }

  // Prevent copy and assignment.
  Layer(const Layer&) = delete;
  Layer& operator=(const Layer&) = delete;

 protected:
  DeviceMatrix input_;
  DeviceMatrix output_;
  DeviceMatrix input_gradient_;
};

#endif  // _CNN_LAYER_H_
