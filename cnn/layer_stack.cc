#include "cnn/layer_stack.h"

LayerStack::LayerStack() {}

void LayerStack::AddLayer(std::shared_ptr<Layer> layer) {
  layers_.push_back(layer);
}

void LayerStack::Forward(const DeviceMatrix& input) {
  DeviceMatrix last_input = input;
  for (std::shared_ptr<Layer> layer : layers_) {
    layer->Forward(last_input);
    last_input = layer->output();
  }
}

void LayerStack::Backward(const DeviceMatrix& output_gradients) {
  DeviceMatrix last_output_gradients = output_gradients;
  for (int i = layers_.size() - 1; i >= 0; i--) {
    layers_[i]->Backward(last_output_gradients);
    last_output_gradients = layers_[i]->input_gradients();
  }
}

void LayerStack::ApplyGradient(float learn_rate) {
  for (std::shared_ptr<Layer> layer : layers_) {
    layer->ApplyGradient(learn_rate);
  }
}
