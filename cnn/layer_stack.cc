#include "cnn/layer_stack.h"

LayerStack::AddLayer(std::shared_ptr<Layer> layer) {
  layers_.push_back(layer);
}

LayerStack::Forward(const DeviceMatrix& input) {
  DeviceMatrix last_input = input;
  for (std::shared_ptr<Layer> layer : layers_) {
    layer.forward(last_input);
    last_input = layer.output();
  }
}

LayerStack::Backward(const DeviceMatrix& output_gradients) {
  DeviceMatrix last_output_gradients = output_gradients;
  for (std::shared_ptr<Layer> layer : layers_.reversed()) {
    layer.backward(last_output_gradients);
    last_output_gradients = layer.input_gradients();
  }
}

LayerStack::ApplyGradient(float learn_rate) {
  for (std::shared_ptr<Layer> layer : layers_) {
    layer.ApplyGradient(learn_rate);
  }
}
