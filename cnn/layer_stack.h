#ifndef _CNN_LAYER_STACK_H_
#define _CNN_LAYER_STACK_H_

#include <vector>
#include <memory>

#include "cnn/layer.h"

class Random;

class LayerStack : public Layer {
 public:
  LayerStack();
  void AddLayer(std::shared_ptr<Layer> layer);
  template <typename T>
  std::shared_ptr<T> GetLayer(int id) {
    // Allow "Python-style" negative indices.
    while (id < 0) id += layers_.size();
    id = id % layers_.size();
    return std::dynamic_pointer_cast<T>(layers_[id]);
  }
  virtual void Print() const;
  virtual void Initialize(Random* random);
  virtual void Forward(const Matrix& input);
  virtual void Backward(const Matrix& output_gradient);
  virtual void ApplyGradient(float learn_rate);
  virtual void Regularize(float lambda);
  virtual bool BeginPhase(Phase phase, int phase_sub_id);
  virtual void EndPhase(Phase phase, int phase_sub_id);

  virtual Matrix output() {
    return layers_.back()->output();
  }
  virtual Matrix input_gradient() {
    return layers_.front()->input_gradient();
  }

 private:
  std::vector<std::shared_ptr<Layer>> layers_;

};

#endif  // _CNN_LAYER_STACK_H_

