#ifndef _CNN_LAYER_H_
#define _CNN_LAYER_H_

#include "linalg/matrix.h"

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
  virtual void Forward(const Matrix& input) = 0;
  virtual void Backward(const Matrix& output_gradient) = 0;
  virtual void ApplyGradient(float /* learn_rate */) {};
  virtual void Regularize(float /* lambda */) {};
  virtual int NumParameters() const { return 0; }

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

  virtual Matrix output() { return output_; }
  virtual Matrix input_gradient() { return input_gradient_; }

  // Prevent copy and assignment.
  Layer(const Layer&) = delete;
  Layer& operator=(const Layer&) = delete;

 protected:
  Matrix input_;
  Matrix output_;
  Matrix input_gradient_;
};

#endif  // _CNN_LAYER_H_
