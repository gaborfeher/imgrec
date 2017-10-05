#include <functional>
#include <iostream>
#include <vector>

#include "cnn/error_layer.h"
#include "cnn/fully_connected_layer.h"
#include "cnn/layer_stack.h"
#include "cnn/model.h"
#include "cnn/sigmoid_layer.h"
#include "linalg/device_matrix.h"

#include "gtest/gtest.h"

void CreateTestCase1(
    DeviceMatrix* training_x,
    DeviceMatrix* training_y,
    DeviceMatrix* test_x,
    DeviceMatrix* test_y) {

  *training_x = DeviceMatrix(8, 3, 1, (float[]) {
      -1,  2, 1,
       0,  1, 1,
       1,  0, 1,
       2, -1, 1,
      -2,  1, 1,
      -1,  0, 1,
       0, -1, 1,
       1, -2, 1,
    }).T();

  *training_y = DeviceMatrix(8, 1, 1, (float[]) {
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
  }).T();

  *test_x = DeviceMatrix(2, 3, 1, (float[]) {
      -1, -1, 1,
       1,  1, 1,
  }).T();

  *test_y = DeviceMatrix(2, 1, 1, (float[]) {
      1,
      0,
  }).T();
}

TEST(LearnTest, FullyConnectedTrain) {
  DeviceMatrix training_x;
  DeviceMatrix training_y;
  DeviceMatrix test_x;
  DeviceMatrix test_y;
  CreateTestCase1(&training_x, &training_y, &test_x, &test_y);

  std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
  stack->AddLayer(std::make_shared<FullyConnectedLayer>(3, 1));
  stack->AddLayer(std::make_shared<SigmoidLayer>());
  Model model(
      stack,
      std::make_shared<ErrorLayer>());

  std::vector<float> training_error;
  model.Train(training_x, training_y, 100, 5, &training_error);
  // for (float err: training_error) {
  //   std::cout << "Training error= " << err << std::endl;
  // }
  float test_error;
  model.Evaluate(test_x, test_y, &test_error);
  EXPECT_LT(test_error, 0.0001);
}

DeviceMatrix ComputeNumericGradients(
    const DeviceMatrix& x0,
    std::function< float (const DeviceMatrix&) > runner
) {

  DeviceMatrix result(x0.rows(), x0.cols(), x0.depth());

  float error0 = runner(x0);
  float delta = 0.0001f;
  for (int k = 0; k < x0.depth(); k++) {
    for (int i = 0; i < x0.rows(); i++) {
      for (int j = 0; j < x0.cols(); j++) {
        DeviceMatrix x1(x0.DeepCopy());
        x1.SetValue(i, j, k, x1.GetValue(i, j, k) + delta);
        float error1 = runner(x1);
        result.SetValue(i, j, k, (error1 - error0) / delta );
      }
    }
  }

  return result;
}

TEST(LearnTest, L2ErrorLayerGradientAt0) {
  ErrorLayer error_layer;
  error_layer.SetExpectedValue(DeviceMatrix(1, 3, 1, (float[]) {-0.5f, 4.2f, -1.0f}));

  // Get gradients with a forward+backward pass (expecting zero gradients here):
  error_layer.Forward(DeviceMatrix(1, 3, 1, (float[]) {-0.5f, 4.2f, -1.0f}));
  error_layer.Backward(DeviceMatrix());
  std::vector<float> grad = error_layer.input_gradients().GetVector();
  EXPECT_FLOAT_EQ(0.0f, grad[0]);
  EXPECT_FLOAT_EQ(0.0f, grad[1]);
  EXPECT_FLOAT_EQ(0.0f, grad[2]);
}

TEST(LearnTest, L2ErrorLayerGradient) {
  ErrorLayer error_layer;
  error_layer.SetExpectedValue(DeviceMatrix(1, 3, 1, (float[]) {-0.5f, 4.2f, -1.0f}));

  // Get gradients with a forward+backward pass:
  error_layer.Forward(DeviceMatrix(1, 3, 1, (float[]) {1.0f, 4.0f, 0.0f}));
  error_layer.Backward(DeviceMatrix());
  std::vector<float> a_grad = error_layer.input_gradients().GetVector();

  // Approximate gradients numerically (at the same position as before):
  DeviceMatrix num_gradients(ComputeNumericGradients(
      DeviceMatrix(1, 3, 1, (float[]) {1.0f, 4.0f, 0.0f}),
      [&error_layer] (const DeviceMatrix& x) -> float {
        error_layer.Forward(x);
        return error_layer.GetError();
      }
  ));
  std::vector<float> n_grad(num_gradients.GetVector());

  // Compare analytically and numerically computed gradients:
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(a_grad[i], n_grad[i], 0.001);
  }
}

TEST(LearnTest, FullyConnectedGradient) {
  DeviceMatrix training_x;
  DeviceMatrix training_y;
  DeviceMatrix test_x;
  DeviceMatrix test_y;
  CreateTestCase1(&training_x, &training_y, &test_x, &test_y);

  std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
  std::shared_ptr<FullyConnectedLayer> fc_layer =
      std::make_shared<FullyConnectedLayer>(3, 1);
  stack->AddLayer(fc_layer);
  // stack->AddLayer(std::make_shared<SigmoidLayer>());
  std::shared_ptr<ErrorLayer> error_layer =
      std::make_shared<ErrorLayer>();
  stack->AddLayer(error_layer);

  error_layer->SetExpectedValue(training_y);

  // Compute gradient the analytical way:
  fc_layer->weights_ =
      DeviceMatrix(1, 3, 1, (float[]) { 4.2, -3.0, 1.7});
  stack->Forward(training_x);
  DeviceMatrix dummy;
  stack->Backward(dummy);
  std::vector<float> a_grad = fc_layer->weights_gradients_.GetVector();

  // Approximate gradient the numerical way:
  std::vector<float> n_grad = ComputeNumericGradients(
      DeviceMatrix(1, 3, 1, (float[]) { 4.2, -3.0, 1.7}),
      [&fc_layer, &stack, training_x, error_layer] (const DeviceMatrix& x) -> float {
        fc_layer->weights_ = x;
        stack->Forward(training_x);
        return error_layer->GetError();
      }).GetVector();
  
  // Compare analytically and numerically computed gradients:
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(a_grad[i], n_grad[i], 0.05);
  }
}
