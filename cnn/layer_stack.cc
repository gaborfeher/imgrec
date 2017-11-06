#include "cnn/layer_stack.h"

#include "util/random.h"

#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>

LayerStack::LayerStack() : phase_last_child_id_(-1) {}

void LayerStack::AddLayer(std::shared_ptr<Layer> layer) {
  layers_.push_back(layer);
}

void LayerStack::Print() const {
  for (std::shared_ptr<Layer> layer : layers_) {
    layer->Print();
  }
}

void LayerStack::Initialize(std::shared_ptr<Random> random) {
  for (std::shared_ptr<Layer> layer : layers_) {
    layer->Initialize(random);
  }
}

void LayerStack::Forward(const Matrix& input) {
  int limit = layers_.size();
  if (phase_last_child_id_ >= 0) {
    limit = phase_last_child_id_ + 1;
  }
  Matrix last_input = input;
  for (int i = 0; i < limit; ++i) {
    layers_[i]->Forward(last_input);
    last_input = layers_[i]->output();
  }
}

void LayerStack::Backward(const Matrix& output_gradient) {
  Matrix last_output_gradient = output_gradient;
  for (int i = layers_.size() - 1; i >= 0; i--) {
    layers_[i]->Backward(last_output_gradient);
    last_output_gradient = layers_[i]->input_gradient();
  }
}

void LayerStack::ApplyGradient(const GradientInfo& info) {
  for (std::shared_ptr<Layer> layer : layers_) {
    layer->ApplyGradient(info);
  }
}

bool LayerStack::OnBeginPhase() {
  phase_last_child_id_ = -1;
  bool result = false;
  for (int i = 0; i < layers_.size(); ++i) {
    if (layers_[i]->BeginPhase(phase(), phase_sub_id())) {
      result = true;
      phase_last_child_id_ = i;
    }
  }
  return result;
}

void LayerStack::OnEndPhase() {
  for (std::shared_ptr<Layer> layer : layers_) {
    layer->EndPhase(phase(), phase_sub_id());
  }
}

int LayerStack::NumParameters() const {
  int total = 0;
  for (std::shared_ptr<Layer> layer : layers_) {
    total += layer->NumParameters();
  }
  return total;
}

void LayerStack::save(cereal::PortableBinaryOutputArchive& ar) const {
  ar(layers_);
}

void LayerStack::load(cereal::PortableBinaryInputArchive& ar) {
  ar(layers_);
}

CEREAL_REGISTER_TYPE(LayerStack);
CEREAL_REGISTER_POLYMORPHIC_RELATION(Layer, LayerStack);

