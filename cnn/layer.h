#ifndef _CNN_LAYER_H_
#define _CNN_LAYER_H_

#include <memory>

#include "linalg/matrix.h"

struct GradientInfo;
class Logger;
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
  virtual void SetLogger(std::shared_ptr<Logger> logger);

  virtual std::string Name() const = 0;
  virtual void Print() const;
  virtual void Initialize(std::shared_ptr<Random> /* generator */) {};
  virtual void Forward(const Matrix& input) = 0;
  virtual void Backward(const Matrix& output_gradient) = 0;
  virtual void ApplyGradient(const GradientInfo&) {}
  virtual int NumParameters() const { return 0; }

  // Signals to the layer that a phase is beginning.
  // For optional phases like PRE_TRAIN_PHASE and POST_TRAIN_PHASE,
  // the return value determines if the phases are needed.
  // (See Model::RunPhase.)
  // Even layers returning false must be able to handle all the
  // phases.
  virtual bool BeginPhase(Phase phase, int phase_sub_id) final;
  virtual void EndPhase(Phase phase, int phase_sub_id) final;

  virtual Matrix output() { return output_; }
  virtual Matrix input_gradient() { return input_gradient_; }

  // Prevent copy and assignment.
  Layer(const Layer&) = delete;
  Layer& operator=(const Layer&) = delete;

 protected:
  Matrix input_;
  Matrix output_;
  Matrix input_gradient_;

  std::shared_ptr<Logger> logger_;

  // Override-able variants of BeginPhase and EndPhase. Use phase()
  // and phase_sub_id() inside them.
  virtual bool OnBeginPhase() {
    return false;
  };
  virtual void OnEndPhase() {};

  Phase phase() { return phase_; }
  int phase_sub_id() { return phase_sub_id_; }

 private:
  // Identifiers of the current Phase.
  Phase phase_;
  int phase_sub_id_;

};

#endif  // _CNN_LAYER_H_
