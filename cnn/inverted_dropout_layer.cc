#include "cnn/inverted_dropout_layer.h"

#include "util/random.h"

#include <cassert>

InvertedDropoutLayer::InvertedDropoutLayer(
    float p,
    std::shared_ptr<Random> random) : p_(p), random_(random) {
}


void InvertedDropoutLayer::Forward(const Matrix& input) {
  input_ = input;
  if (phase_ == TRAIN_PHASE) {
    if (mask_.rows() != input.rows() ||
        mask_.cols() != input.cols() ||
        mask_.depth() != input.depth()) {
      mask_ = Matrix(input.rows(), input.cols(), input.depth());
    }
    mask_.InvertedDropoutFill(random_.get(), p_);
    output_ = input.ElementwiseMultiply(mask_);
  } else {
    output_ = input;
  }
}

void InvertedDropoutLayer::Backward(const Matrix& output_gradient) {
  if (phase_ == TRAIN_PHASE) {
    input_gradient_ = output_gradient.ElementwiseMultiply(mask_);
  } else {
    input_gradient_ = output_gradient;
  }
}

bool InvertedDropoutLayer::BeginPhase(Phase phase, int phase_sub_id) {
  phase_ = phase;
  phase_sub_id_ = phase_sub_id;
  return false;
}

void InvertedDropoutLayer::EndPhase(Phase phase, int phase_sub_id) {
  assert(phase_sub_id == phase_sub_id_);
  assert(phase == phase_);
  phase_ = NONE;
  phase_sub_id_ = -1;
}
