#include <memory>

#include "cnn/layer.h"
#include "cnn/inverted_dropout_layer.h"
#include "linalg/matrix.h"
#include "util/random.h"

#include "gtest/gtest.h"

TEST(InvertedDropoutLayerTest, SmokeTest) {
  InvertedDropoutLayer layer(0.5, std::make_shared<Random>(42));
  layer.BeginPhase(Layer::TRAIN_PHASE, 0);
  layer.Forward(Matrix(2, 3, 2, {
      1, 1, 1,
      1, 1, 1,
      2, 2, 2,
      2, 2, 2,
  }));
}
