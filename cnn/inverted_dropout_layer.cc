#include "cnn/inverted_dropout_layer.h"

#include "util/random.h"

#include <cassert>

InvertedDropoutLayer::InvertedDropoutLayer(
    int num_neurons,
    bool layered,
    float p,
    std::shared_ptr<Random> random) :
        BiasLikeLayer(num_neurons, layered),
        p_(p),
        random_(random) {
}


void InvertedDropoutLayer::Forward(const Matrix& input) {
  if (phase() == TRAIN_PHASE) {
    Matrix rands = Matrix::MakeInvertedDropoutMask(
        layered_, num_neurons_, p_, random_.get());
    mask_ = rands.Repeat(layered_, input);
    output_ = input.ElementwiseMultiply(mask_);
  } else {
    output_ = input;
  }
}

void InvertedDropoutLayer::Backward(const Matrix& output_gradient) {
  if (phase() == TRAIN_PHASE) {
    input_gradient_ = output_gradient.ElementwiseMultiply(mask_);
  } else {
    input_gradient_ = output_gradient;
  }
}

