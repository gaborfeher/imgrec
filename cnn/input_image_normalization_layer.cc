#include "cnn/input_image_normalization_layer.h"

#include <iostream>
#include <cassert>

#include "cereal/archives/portable_binary.hpp"
#include "cereal/types/memory.hpp"
#include "cereal/types/polymorphic.hpp"

#include "cnn/layer.h"
#include "linalg/matrix.h"

InputImageNormalizationLayer::InputImageNormalizationLayer(int rows, int cols, int depth) :
    num_samples_(0),
    mean_(rows, cols, depth) {}

std::string InputImageNormalizationLayer::Name() const {
  return "InputImageNormalizationLayer";
}

void InputImageNormalizationLayer::Print() const {
  std::cout << Name() << ":" << std::endl;
  mean_.Print();
}

void InputImageNormalizationLayer::Forward(const Matrix& input) {
  input_ = input;
  if (phase() == PRE_TRAIN_PHASE && phase_sub_id() == 0) {
    output_ = input;
    mean_ = mean_.Add(
        input_.PerLayerSum(mean_.depth()));
    num_samples_ += input_.depth() / mean_.depth();
  } else {
    assert(input_.depth() % mean_.depth() == 0);
    output_ = input_.Add(
        mean_.PerLayerRepeat(input_.depth() / mean_.depth()));
  }
}

void InputImageNormalizationLayer::Backward(const Matrix& output_gradient) {
  input_gradient_ = output_gradient;
}

bool InputImageNormalizationLayer::OnBeginPhase() {
  if (phase() == PRE_TRAIN_PHASE) {
    if (phase_sub_id() == 0) {
      num_samples_ = 0;
      mean_.Fill(0);
      return true;
    }
  }
  return false;
}

void InputImageNormalizationLayer::OnEndPhase() {
  if (phase() == PRE_TRAIN_PHASE) {
    if (phase_sub_id() == 0) {
      mean_ = mean_.Multiply(-1.0f / num_samples_);
    }
  }
}

void InputImageNormalizationLayer::save(cereal::PortableBinaryOutputArchive& ar) const {
  ar(mean_, num_samples_);
}

void InputImageNormalizationLayer::load(cereal::PortableBinaryInputArchive& ar) {
  ar(mean_, num_samples_);
}

CEREAL_REGISTER_TYPE(InputImageNormalizationLayer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(Layer, InputImageNormalizationLayer);

