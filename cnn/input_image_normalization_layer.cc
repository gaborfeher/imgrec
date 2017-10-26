#include "cnn/input_image_normalization_layer.h"

#include <iostream>
#include <cassert>

#include "cnn/layer.h"
#include "linalg/matrix.h"

InputImageNormalizationLayer::InputImageNormalizationLayer(int rows, int cols, int depth) :
    num_samples_(0),
    mean_(rows, cols, depth) {}

void InputImageNormalizationLayer::Print() const {
  std::cout << "Input Image Normalization Layer" << std::endl;
  mean_.Print();
}

void InputImageNormalizationLayer::Forward(const Matrix& input) {
  input_ = input;
  if (phase_ == PRE_TRAIN_PHASE && phase_sub_id_ == 0) {
    output_ = input;
    mean_ = mean_.Add(
        input_
            .PerLayerSum(mean_.depth())
            .Multiply(1.0f * mean_.depth() / input_.depth()));
    num_samples_ += 1;
  } else {
    assert(input_.depth() % mean_.depth() == 0);
    output_ = input_.Add(
        mean_.PerLayerRepeat(input_.depth() / mean_.depth()));
  }
}

void InputImageNormalizationLayer::Backward(const Matrix& output_gradient) {
  input_gradient_ = output_gradient;
}

bool InputImageNormalizationLayer::BeginPhase(Phase phase, int phase_sub_id) {
  phase_ = phase;
  phase_sub_id_ = phase_sub_id;
  if (phase == PRE_TRAIN_PHASE) {
    if (phase_sub_id_ == 0) {
      num_samples_ = 0;
      mean_.Fill(0);
      return true;
    }
  }
  return false;
}

void InputImageNormalizationLayer::EndPhase(Phase phase, int phase_sub_id) {
  assert(phase_sub_id == phase_sub_id_);
  assert(phase == phase_);
  if (phase_ == PRE_TRAIN_PHASE) {
    if (phase_sub_id_ == 0) {
      mean_ = mean_.Multiply(-1.0f / num_samples_);
    }
  }
  phase_ = NONE;
}
