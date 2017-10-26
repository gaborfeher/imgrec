#include "gtest/gtest.h"

#include "cnn/layer.h"
#include "cnn/input_image_normalization_layer.h"
#include "linalg/matrix_test_util.h"

TEST(InputImageNormalizationLayerTest, MeanTest) {
  // 2 layers per image, 2 images per batch, 2 batches:
  Matrix batch1(2, 3, 4, (float[]) {
    1, 1, 1,  // img1 layer1
    2, 2, 2,
    2, 2, 2,  // img1 layer2
    1, 1, 1,

    1, 2, 3,  // img2 layer1
    0, 0, 0,
    -1, -1, 1,  // img2 layer2
    0, 0, 0,
  });
  Matrix batch2(2, 3, 4, (float[]) {
    -1, 0, 1,  // img1 layer1
    0, -1, 2,
    -2, -2, 2,  // img1 layer2
    1, 2, 3,

    3, 2, 1,  // img2 layer1
    -1, 1, -1,
    2, 2, 2,  // img2 layer2
    -2, -2, -2,
  });
  // A batch with 3 images to test inference
  Matrix infer_batch(2, 3, 6, (float[]) {
    // Image1: we expect to see negative mean
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    // Image2:
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    // Image3:
    -1, 1, -1,
    1, -1, 1,
    -1, 0, 1,
    2, 2, 2,
  });

  InputImageNormalizationLayer layer(2, 3, 2);
  EXPECT_TRUE(layer.BeginPhase(Layer::PRE_TRAIN_PHASE, 0));
  layer.Forward(batch1);
  layer.Forward(batch2);
  layer.EndPhase(Layer::PRE_TRAIN_PHASE, 0);
  layer.BeginPhase(Layer::INFER_PHASE, 0);
  layer.Forward(infer_batch);
  layer.EndPhase(Layer::INFER_PHASE, 0);
  // Check that the global mean was correctly subtracted from
  // infer_batch.
  ExpectMatrixEquals(
      Matrix(2, 3, 6, (float[]) {
          -1, -1.25, -1.5,
          -0.25, -0.5, -0.75,
          -0.25, -0.25, -1.75,
          0, -0.25, -0.5,

          1 - 1, 1 - 1.25, 1 - 1.5,
          1 - 0.25, 1 - 0.5, 1 - 0.75,
          1 - 0.25, 1 - 0.25, 1 - 1.75,
          1 - 0, 1 - 0.25, 1 - 0.5,

          -1 - 1, 1 - 1.25, -1 - 1.5,
          1 - 0.25, -1 - 0.5, 1 - 0.75,
          -1 - 0.25, 0 - 0.25, 1 - 1.75,
          2 - 0, 2 - 0.25, 2 - 0.5,
      }),
      layer.output());
}

