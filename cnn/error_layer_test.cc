#include <functional>
#include <iostream>
#include <vector>

#include "cnn/l2_error_layer.h"
#include "cnn/layer_test_base.h"
#include "cnn/softmax_error_layer.h"
#include "linalg/device_matrix.h"

#include "gtest/gtest.h"

TEST(ErrorLayerTest, L2GradientAt0) {
  L2ErrorLayer error_layer;
  error_layer.SetExpectedValue(DeviceMatrix(1, 3, 1, (float[]) {-0.5f, 4.2f, -1.0f}));

  // Get gradients with a forward+backward pass (expecting zero gradients here):
  error_layer.Forward(DeviceMatrix(1, 3, 1, (float[]) {-0.5f, 4.2f, -1.0f}));
  error_layer.Backward(DeviceMatrix());
  std::vector<float> grad = error_layer.input_gradients().GetVector();
  EXPECT_FLOAT_EQ(0.0f, grad[0]);
  EXPECT_FLOAT_EQ(0.0f, grad[1]);
  EXPECT_FLOAT_EQ(0.0f, grad[2]);
}

TEST(ErrorLayerTest, L2Gradient) {
  L2ErrorLayer error_layer;
  error_layer.SetExpectedValue(DeviceMatrix(1, 3, 1, (float[]) {-0.5f, 4.2f, -1.0f}));
  DeviceMatrix input(1, 3, 1, (float[]) {1.0f, 4.0f, 0.0f});

  // Get gradients with a forward+backward pass:
  error_layer.Forward(input);
  error_layer.Backward(DeviceMatrix());
  DeviceMatrix a_grad = error_layer.input_gradients();

  // Approximate gradients numerically (at the same position as before):
  DeviceMatrix n_grad = ComputeNumericGradients(
      input,
      [&error_layer] (const DeviceMatrix& x) -> float {
        error_layer.Forward(x);
        return error_layer.GetError();
      }
  );

  // Compare analytically and numerically computed gradients:
  ExpectMatrixEquals(a_grad, n_grad, 0.001f, 5);
}

TEST(ErrorLayerTest, SoftmaxGradient) {
  SoftmaxErrorLayer error_layer;
  error_layer.SetExpectedValue(DeviceMatrix(1, 3, 1, (float[]) {
      0.0f, 1.0f, 2.0f}));
  DeviceMatrix input(3, 3, 1, (float[]) {
      -2.0f,  1.0f, 1.0f,
       1.0f, -1.0f, 2.0f,
       0.5f, -2.0f, 3.0f
  });

  // Get gradients with a forward+backward pass:
  error_layer.Forward(input);
  error_layer.Backward(DeviceMatrix());
  DeviceMatrix a_grad = error_layer.input_gradients();

  // Approximate gradients numerically (at the same position as before):
  DeviceMatrix n_grad = ComputeNumericGradients(
      input,
      [&error_layer] (const DeviceMatrix& x) -> float {
        error_layer.Forward(x);
        return error_layer.GetError();
      }
  );

  // Compare analytically and numerically computed gradients:
  ExpectMatrixEquals(a_grad, n_grad, 0.001f, 5);
}

