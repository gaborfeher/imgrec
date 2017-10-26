#include "gtest/gtest.h"

#include "cnn/pooling_layer.h"
#include "cnn/l2_error_layer.h"
#include "cnn/layer_stack.h"
#include "cnn/layer_test_base.h"
#include "linalg/matrix.h"
#include "linalg/matrix_test_util.h"

TEST(PoolingLayerTest, Forward_Backward_ManualCheck) {
  Matrix training_x(4, 4, 1, (float[]) {
      1, 2, 4, 1,
      3, 4, 2, 3,
      3, 4, 2, 3,
      1, 2, 4, 1,
  });
  Matrix training_y(2, 2, 1, (float[]) {
      11, 12,
      13, 14,
  });

  PoolingLayer layer(2, 2);
  layer.Forward(training_x);
  ExpectMatrixEquals(
      Matrix(2, 2, 1, (float[]) {
          4, 4,
          4, 4,
      }),
      layer.output());
  layer.Backward(training_y);

  ExpectMatrixEquals(
      Matrix(4, 4, 1, (float[]) {
          0, 0, 12, 0,
          0, 11, 0, 0,
          0, 13, 0, 0,
          0, 0, 14, 0,
      }),
      layer.input_gradient());
}

TEST(PoolingLayerTest, Gradient_AutoCheck) {
  Matrix training_x(4, 4, 2, (float[]) {
      1, 2, 4, 1,
      3, 4, 2, 3,
      3, 4, 2, 3,
      1, 2, 4, 1,

      -2, 2, -1, 3,
      -2, 3, 2, -1,
      -2, -1, 1, -1,
      -2, -2, -2, 2,
  });
  Matrix training_y(2, 2, 2, (float[]) {
      4, 5,
      5, 4,
      2, 3,
      3, 2,
  });

  std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
  stack->AddLayer(std::make_shared<PoolingLayer>(2, 2));
  stack->AddLayer(std::make_shared<L2ErrorLayer>());
  stack->GetLayer<ErrorLayer>(1)->SetExpectedValue(training_y);

  InputGradientCheck(
      stack,
      training_x,
      0.01f,
      1.0f);
}
