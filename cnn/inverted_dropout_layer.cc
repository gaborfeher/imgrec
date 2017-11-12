#include "cnn/inverted_dropout_layer.h"

#include <cassert>

#include "cereal/archives/portable_binary.hpp"
#include "cereal/types/memory.hpp"
#include "cereal/types/polymorphic.hpp"

#include "util/random.h"

InvertedDropoutLayer::InvertedDropoutLayer(
    int num_neurons,
    bool layered,
    float p,
    std::shared_ptr<Random> random) :
        BiasLikeLayer(num_neurons, layered),
        p_(p),
        random_(random) {
}

std::string InvertedDropoutLayer::Name() const {
  return "InvertedDropoutLayer";
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

void InvertedDropoutLayer::save(cereal::PortableBinaryOutputArchive& ar) const {
  ar(num_neurons_, layered_, p_);
}

void InvertedDropoutLayer::load(cereal::PortableBinaryInputArchive& ar) {
  ar(num_neurons_, layered_, p_);
}

CEREAL_REGISTER_TYPE(InvertedDropoutLayer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(Layer, InvertedDropoutLayer);
