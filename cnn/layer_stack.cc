#include "cnn/layer_stack.h"

#include "util/random.h"

LayerStack::LayerStack() {}

void LayerStack::AddLayer(std::shared_ptr<Layer> layer) {
  layers_.push_back(layer);
}

void LayerStack::Print() const {
  for (std::shared_ptr<Layer> layer : layers_) {
    layer->Print();
  }
}

void LayerStack::Initialize(Random* random) {
  for (std::shared_ptr<Layer> layer : layers_) {
    layer->Initialize(random);
  }
}

void LayerStack::Forward(const Matrix& input) {
  Matrix last_input = input;
  for (std::shared_ptr<Layer> layer : layers_) {
    layer->Forward(last_input);
    last_input = layer->output();
  }
}

void LayerStack::Backward(const Matrix& output_gradient) {
  Matrix last_output_gradient = output_gradient;
  for (int i = layers_.size() - 1; i >= 0; i--) {
    layers_[i]->Backward(last_output_gradient);
    last_output_gradient = layers_[i]->input_gradient();
  }
}

void LayerStack::ApplyGradient(float learn_rate) {
  for (std::shared_ptr<Layer> layer : layers_) {
    layer->ApplyGradient(learn_rate);
  }
}

void LayerStack::Regularize(float lambda) {
  for (std::shared_ptr<Layer> layer : layers_) {
    layer->Regularize(lambda);
  }
}

bool LayerStack::BeginPhase(Phase phase, int phase_sub_id) {
  bool result = false;
  for (std::shared_ptr<Layer> layer : layers_) {
    if (layer->BeginPhase(phase, phase_sub_id)) {
      result = true;
    }
  }
  return result;
}

void LayerStack::EndPhase(Phase phase, int phase_sub_id) {
  for (std::shared_ptr<Layer> layer : layers_) {
    layer->EndPhase(phase, phase_sub_id);
  }
}

