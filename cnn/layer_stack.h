#ifndef _CNN_LAYER_STACK_H_
#define _CNN_LAYER_STACK_H_

#include <vector>
#include <memory>

#include "cnn/layer.h"

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
  virtual void Forward(const DeviceMatrix& input);
  virtual void Backward(const DeviceMatrix& output_gradients);
  virtual void ApplyGradient(float learn_rate);

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

