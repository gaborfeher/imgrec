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

  // Like AddLayer, but creates a new layer object. The arguments
  // of this function are passed to the layer's constructor.
  // (This works the same way as std::make_shared<>.)
  template <typename T, class ... Args>
  void AddLayer(Args && ... args) {
    AddLayer(std::make_shared<T>(args...));
  }

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
  virtual void ApplyGradient(float learn_rate, float lambda);
  virtual bool OnBeginPhase();
  virtual void OnEndPhase();
  virtual int NumParameters() const;

  virtual Matrix output() {
    return layers_.back()->output();
  }
  virtual Matrix input_gradient() {
    return layers_.front()->input_gradient();
  }

 private:
  std::vector<std::shared_ptr<Layer>> layers_;
  // When not all children return true for BeginPhase, then
  // we only need to run forward passes until the last child
  // returning true. (If all returned false this is disabled.)
  int phase_last_child_id_;

};

#endif  // _CNN_LAYER_STACK_H_

