#include <iostream>

#include "cnn/convolutional_layer.h"
#include "cnn/fully_connected_layer.h"
#include "cnn/l2_error_layer.h"
#include "cnn/layer_stack.h"
#include "cnn/layer_test_base.h"
#include "cnn/reshape_layer.h"
#include "cnn/nonlinearity_layer.h"
#include "infra/data_set.h"
#include "infra/model.h"
#include "linalg/device_matrix.h"
#include "linalg/matrix_test_util.h"

#include "gtest/gtest.h"

// For fully connected
std::shared_ptr<InMemoryDataSet> CreateTestCase1_TrainingData() {
  return std::make_shared<InMemoryDataSet>(
      8,
      DeviceMatrix(8, 3, 1, (float[]) {
        -1,  2, 1,
         0,  1, 1,
         1,  0, 1,
         2, -1, 1,
        -2,  1, 1,
        -1,  0, 1,
         0, -1, 1,
         1, -2, 1,
      }).T(),
      DeviceMatrix(8, 1, 1, (float[]) {
          0,
          0,
          0,
          0,
          1,
          1,
          1,
          1,
      }).T());
}

std::shared_ptr<InMemoryDataSet> CreateTestCase1_TestData() {
  return std::make_shared<InMemoryDataSet>(
      2,
      DeviceMatrix(2, 3, 1, (float[]) {
          -1, -1, 1,
           1,  1, 1,
      }).T(),
      DeviceMatrix(2, 1, 1, (float[]) {
          1,
          0,
      }).T());
}

TEST(LearnTest, FullyConnectedTrain) {
  std::shared_ptr<InMemoryDataSet> training = CreateTestCase1_TrainingData();
  std::shared_ptr<InMemoryDataSet> test = CreateTestCase1_TestData();

  std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
  std::shared_ptr<L2ErrorLayer> error_layer = std::make_shared<L2ErrorLayer>();

  stack->AddLayer(std::make_shared<FullyConnectedLayer>(3, 1));
  stack->AddLayer(std::make_shared<NonlinearityLayer>(
      ::activation_functions::Sigmoid()));
  stack->AddLayer(error_layer);
  Model model(stack, 42);

  model.Train(
      *training,
      100,
      40,
      0);
  float test_error;
  float test_accuracy;
  model.Evaluate(
      *test,
      &test_error,
      &test_accuracy);
  EXPECT_LT(test_error, 0.0001);
}

TEST(LearnTest, FullyConnectedLayerWeightGradient) {
  std::shared_ptr<InMemoryDataSet> training = CreateTestCase1_TrainingData();
  DeviceMatrix training_x = training->GetBatchInput(0);
  DeviceMatrix training_y = training->GetBatchOutput(0);

  std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
  std::shared_ptr<FullyConnectedLayer> fc_layer =
      std::make_shared<FullyConnectedLayer>(3, 1);
  stack->AddLayer(fc_layer);
  stack->AddLayer(std::make_shared<NonlinearityLayer>(
      ::activation_functions::Sigmoid()));
  std::shared_ptr<L2ErrorLayer> error_layer =
      std::make_shared<L2ErrorLayer>();
  stack->AddLayer(error_layer);

  error_layer->SetExpectedValue(training_y);

  // Compute gradient the analytical way:
  DeviceMatrix weights(1, 3 + 1, 1, (float[]) { 4.2, -3.0, 1.7, -1.0});
  fc_layer->weights_ = weights;
  stack->Forward(training_x);
  DeviceMatrix dummy;
  stack->Backward(dummy);
  DeviceMatrix a_grad = fc_layer->weights_gradients_;

  // Approximate gradient the numerical way:
  DeviceMatrix n_grad = ComputeNumericGradients(
      weights,
      [&fc_layer, &stack, training_x, error_layer] (const DeviceMatrix& x) -> float {
        fc_layer->weights_ = x;
        stack->Forward(training_x);
        return error_layer->GetError();
      });

  // Compare analytically and numerically computed gradients:
  ExpectMatrixEquals(a_grad, n_grad, 0.0001f, 1);
}

TEST(LearnTest, FullyConnectedLayerInputGradient) {
  std::shared_ptr<InMemoryDataSet> training = CreateTestCase1_TrainingData();
  DeviceMatrix training_x = training->GetBatchInput(0);
  DeviceMatrix training_y = training->GetBatchOutput(0);

  std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
  std::shared_ptr<FullyConnectedLayer> fc_layer =
      std::make_shared<FullyConnectedLayer>(3, 1);
  stack->AddLayer(fc_layer);
  stack->AddLayer(std::make_shared<NonlinearityLayer>(
      activation_functions::Sigmoid()));
  std::shared_ptr<L2ErrorLayer> error_layer =
      std::make_shared<L2ErrorLayer>();
  stack->AddLayer(error_layer);

  error_layer->SetExpectedValue(training_y);
  DeviceMatrix weights(1, 3 + 1, 1, (float[]) { 4.2, -3.0, 1.7, -1.0});
  fc_layer->weights_ = weights;

  // Compute gradient the analytical way:
  stack->Forward(training_x);
  DeviceMatrix dummy;
  stack->Backward(dummy);
  DeviceMatrix a_grad = fc_layer->input_gradients();

  // Approximate gradient the numerical way:
  DeviceMatrix n_grad = ComputeNumericGradients(
      training_x,
      [&stack, error_layer] (const DeviceMatrix& x) -> float {
        stack->Forward(x);
        return error_layer->GetError();
      });
  
  // Compare analytically and numerically computed gradients:
  ExpectMatrixEquals(a_grad, n_grad, 0.01f, -1  /* :( */);
}

