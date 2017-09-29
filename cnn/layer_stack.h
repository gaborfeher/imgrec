#ifndef _CNN_LAYER_STACK_H_
#define _CNN_LAYER_STACK_H_

class LayerStack : public Layer {
 public:
  void AddLayer(std::shared_ptr<Layer> layer);
  virtual void forward(const DeviceMatrix& input);
  virtual void backward(const DeviceMatrox& output_gradients);

  virtual DeviceMatrix output() { return layers_.last().output(); }
  virtual DeviceMatrix input_gradients() { return layers_.first().input_gradients(); }

 private:
  vector<std::shared_ptr<Layer>> layers_;

}

#endif  // _CNN_LAYER_STACK_H_

