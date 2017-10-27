#include <functional>
#include <iostream>
#include <vector>

#include "cnn/l2_error_layer.h"
#include "cnn/layer_stack.h"
#include "cnn/layer_test_base.h"
#include "cnn/softmax_error_layer.h"
#include "linalg/matrix.h"
#include "linalg/matrix_test_util.h"

#include "gtest/gtest.h"

TEST(ErrorLayerTest, L2GradientAt0) {
  L2ErrorLayer error_layer;
  error_layer.SetExpectedValue(Matrix(1, 3, 1, {-0.5f, 4.2f, -1.0f}));

  // Get gradient with a forward+backward pass (expecting zero gradient here):
  error_layer.Forward(Matrix(1, 3, 1, {-0.5f, 4.2f, -1.0f}));
  error_layer.Backward(Matrix());
  ExpectMatrixEquals(
      Matrix(1, 3, 1, { 0.0f, 0.0f, 0.0f }),
      error_layer.input_gradient());
}

TEST(ErrorLayerTest, L2Gradient) {
  std::shared_ptr<L2ErrorLayer> error_layer =
      std::make_shared<L2ErrorLayer>();
  error_layer->SetExpectedValue(Matrix(1, 3, 1, {-0.5f, 4.2f, -1.0f}));
  std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
  stack->AddLayer(error_layer);

  Matrix input(1, 3, 1,  {1.0f, 4.0f, 0.0f});

  InputGradientCheck(
      stack,
      input,
      0.001f,
      0.1f);
}

TEST(ErrorLayerTest, SoftmaxGradient) {
  std::shared_ptr<SoftmaxErrorLayer> error_layer =
      std::make_shared<SoftmaxErrorLayer>();
  error_layer->SetExpectedValue(Matrix(1, 3, 1, {
      0.0f, 1.0f, 2.0f}));
  std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
  stack->AddLayer(error_layer);

  Matrix input(3, 3, 1, {
      -2.0f,  1.0f, 1.0f,
       1.0f, -1.0f, 2.0f,
       0.5f, -2.0f, 3.0f
  });

  InputGradientCheck(
      stack,
      input,
      0.001f,
      1.0f);
}

