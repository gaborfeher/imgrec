#include <functional>
#include <iostream>
#include <vector>

#include "cnn/l2_error_layer.h"
#include "cnn/layer_stack.h"
#include "cnn/layer_test_base.h"
#include "cnn/softmax_error_layer.h"
#include "linalg/device_matrix.h"
#include "linalg/matrix_test_util.h"

#include "gtest/gtest.h"

TEST(ErrorLayerTest, L2GradientAt0) {
  L2ErrorLayer error_layer;
  error_layer.SetExpectedValue(DeviceMatrix(1, 3, 1, (float[]) {-0.5f, 4.2f, -1.0f}));

  // Get gradient with a forward+backward pass (expecting zero gradient here):
  error_layer.Forward(DeviceMatrix(1, 3, 1, (float[]) {-0.5f, 4.2f, -1.0f}));
  error_layer.Backward(DeviceMatrix());
  std::vector<float> grad = error_layer.input_gradient().GetVector();
  EXPECT_FLOAT_EQ(0.0f, grad[0]);
  EXPECT_FLOAT_EQ(0.0f, grad[1]);
  EXPECT_FLOAT_EQ(0.0f, grad[2]);
}

TEST(ErrorLayerTest, L2Gradient) {
  std::shared_ptr<L2ErrorLayer> error_layer =
      std::make_shared<L2ErrorLayer>();
  error_layer->SetExpectedValue(DeviceMatrix(1, 3, 1, (float[]) {-0.5f, 4.2f, -1.0f}));
  std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
  stack->AddLayer(error_layer);

  DeviceMatrix input(1, 3, 1, (float[]) {1.0f, 4.0f, 0.0f});

  InputGradientCheck(
      stack,
      input,
      0.001f,
      0.1f);
}

TEST(ErrorLayerTest, SoftmaxGradient) {
  std::shared_ptr<SoftmaxErrorLayer> error_layer =
      std::make_shared<SoftmaxErrorLayer>();
  error_layer->SetExpectedValue(DeviceMatrix(1, 3, 1, (float[]) {
      0.0f, 1.0f, 2.0f}));
  std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
  stack->AddLayer(error_layer);

  DeviceMatrix input(3, 3, 1, (float[]) {
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

