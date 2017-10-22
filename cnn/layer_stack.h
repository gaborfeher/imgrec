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
  virtual void Forward(const DeviceMatrix& input);
  virtual void Backward(const DeviceMatrix& output_gradient);
  virtual void ApplyGradient(float learn_rate);
  virtual void Regularize(float lambda);
  virtual bool BeginPhase(Phase phase, int phase_sub_id);
  virtual void EndPhase(Phase phase, int phase_sub_id);

  virtual DeviceMatrix output() {
    return layers_.back()->output();
  }
  virtual DeviceMatrix input_gradients() {
    return layers_.front()->input_gradients();
  }

 private:
  std::vector<std::shared_ptr<Layer>> layers_;

};

#endif  // _CNN_LAYER_STACK_H_

