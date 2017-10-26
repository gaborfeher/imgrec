#include <iostream>

#include "cnn/bias_layer.h"
#include "cnn/convolutional_layer.h"
#include "cnn/fully_connected_layer.h"
#include "cnn/l2_error_layer.h"
#include "cnn/layer_stack.h"
#include "cnn/layer_test_base.h"
#include "cnn/reshape_layer.h"
#include "cnn/nonlinearity_layer.h"
#include "infra/data_set.h"
#include "infra/model.h"
#include "linalg/matrix.h"
#include "linalg/matrix_test_util.h"

#include "gtest/gtest.h"

// For fully connected
std::shared_ptr<InMemoryDataSet> CreateTestCase1_TrainingData() {
  return std::make_shared<InMemoryDataSet>(
      8,
      Matrix(8, 2, 1, (float[]) {
        -1,  2,
         0,  1,
         1,  0,
         2, -1,
        -2,  1,
        -1,  0,
         0, -1,
         1, -2,
      }).T(),
      Matrix(8, 1, 1, (float[]) {
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
      Matrix(2, 2, 1, (float[]) {
          -1, -1,
           1,  1,
      }).T(),
      Matrix(2, 1, 1, (float[]) {
          1,
          0,
      }).T());
}

TEST(LearnTest, FullyConnectedTrain) {
  std::shared_ptr<InMemoryDataSet> training = CreateTestCase1_TrainingData();
  std::shared_ptr<InMemoryDataSet> test = CreateTestCase1_TestData();

  std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
  std::shared_ptr<L2ErrorLayer> error_layer = std::make_shared<L2ErrorLayer>();

  stack->AddLayer(std::make_shared<FullyConnectedLayer>(2, 1));
  stack->AddLayer(std::make_shared<BiasLayer>(1, false));
  stack->AddLayer(std::make_shared<NonlinearityLayer>(
      ::activation_functions::Sigmoid()));
  stack->AddLayer(error_layer);
  Model model(stack, 42, false);
  model.Train(
      *training,
      1000,
      1,
      0.0);
  // stack->Print();

  float training_error;
  float training_accuracy;
  model.Evaluate(
      *training,
      &training_error,
      &training_accuracy);
  EXPECT_LT(training_error, 0.0001);

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
  Matrix training_x = training->GetBatchInput(0);
  Matrix training_y = training->GetBatchOutput(0);

  std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
  std::shared_ptr<FullyConnectedLayer> fc_layer =
      std::make_shared<FullyConnectedLayer>(2, 1);
  stack->AddLayer(fc_layer);
  stack->AddLayer(std::make_shared<BiasLayer>(1, false));
  stack->AddLayer(std::make_shared<NonlinearityLayer>(
      ::activation_functions::Sigmoid()));
  std::shared_ptr<L2ErrorLayer> error_layer =
      std::make_shared<L2ErrorLayer>();
  stack->AddLayer(error_layer);

  error_layer->SetExpectedValue(training_y);

  Matrix weights(1, 2, 1, (float[]) { 4.2, -3.0 });
  ParameterGradientCheck(
      stack,
      training_x,
      weights,
      [&fc_layer] (const Matrix& p) -> void {
        fc_layer->weights_ = p;
      },
      [fc_layer] () -> Matrix {
        return fc_layer->weights_gradient_;
      },
      0.001f,
      1);
}

TEST(LearnTest, FullyConnectedLayerInputGradient) {
  std::shared_ptr<InMemoryDataSet> training = CreateTestCase1_TrainingData();
  Matrix training_x = training->GetBatchInput(0);
  Matrix training_y = training->GetBatchOutput(0);

  std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
  std::shared_ptr<FullyConnectedLayer> fc_layer =
      std::make_shared<FullyConnectedLayer>(2, 1);
  stack->AddLayer(fc_layer);
  stack->AddLayer(std::make_shared<BiasLayer>(1, false));
  stack->AddLayer(std::make_shared<NonlinearityLayer>(
      activation_functions::Sigmoid()));
  std::shared_ptr<L2ErrorLayer> error_layer =
      std::make_shared<L2ErrorLayer>();
  stack->AddLayer(error_layer);

  error_layer->SetExpectedValue(training_y);
  Matrix weights(1, 2, 1, (float[]) { 4.2, -3.0 });
  fc_layer->weights_ = weights;

  InputGradientCheck(
      stack,
      training_x,
      0.001f,
      7);
}

