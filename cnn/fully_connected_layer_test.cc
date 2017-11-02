#include <iostream>

#include "cnn/batch_normalization_layer.h"
#include "cnn/bias_layer.h"
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
#include "util/random.h"

#include "gtest/gtest.h"

// For fully connected
std::shared_ptr<InMemoryDataSet> CreateTestCase1_TrainingData() {
  return std::make_shared<InMemoryDataSet>(
      8,
      Matrix(8, 2, 1, {
        -1,  2,
         0,  1,
         1,  0,
         2, -1,
        -2,  1,
        -1,  0,
         0, -1,
         1, -2,
      }).T(),
      Matrix(8, 1, 1,  {
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
      Matrix(2, 2, 1,  {
          -1, -1,
           1,  1,
      }).T(),
      Matrix(2, 1, 1,  {
          1,
          0,
      }).T());
}

TEST(FullyConnectedLayerTest, Train_L2) {
  std::shared_ptr<InMemoryDataSet> training = CreateTestCase1_TrainingData();
  std::shared_ptr<InMemoryDataSet> test = CreateTestCase1_TestData();

  std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
  stack->AddLayer<FullyConnectedLayer>(2, 1);
  stack->AddLayer<BiasLayer>(1, false);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::Sigmoid());
  stack->AddLayer<L2ErrorLayer>();

  Model model(stack, std::make_shared<Random>(42));
  model.Train(
      *training,
      1000,
      GradientInfo(1, 0, GradientInfo::ADAM));
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

TEST(FullyConnectedLayerTest, Train_BatchNorm) {
  std::shared_ptr<InMemoryDataSet> training = CreateTestCase1_TrainingData();
  std::shared_ptr<InMemoryDataSet> test = CreateTestCase1_TestData();

  std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
  stack->AddLayer<FullyConnectedLayer>(2, 1);
  stack->AddLayer<BatchNormalizationLayer>(1, false);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::Sigmoid());
  stack->AddLayer<L2ErrorLayer>();

  Model model(stack, std::make_shared<Random>(42));
  model.Train(
      *training,
      1000,
      GradientInfo(1, 0, GradientInfo::ADAM));
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

TEST(FullyConnectedLayerTest, WeightGradient) {
  std::shared_ptr<InMemoryDataSet> training = CreateTestCase1_TrainingData();
  Matrix training_x = training->GetBatchInput(0);
  Matrix training_y = training->GetBatchOutput(0);

  std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
  std::shared_ptr<FullyConnectedLayer> fc_layer =
      std::make_shared<FullyConnectedLayer>(2, 1);
  stack->AddLayer(fc_layer);
  stack->AddLayer<BiasLayer>(1, false);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::Sigmoid());
  stack->AddLayer<L2ErrorLayer>();
  stack->GetLayer<L2ErrorLayer>(-1)->SetExpectedValue(training_y);

  Matrix weights(1, 2, 1,  { 4.2, -3.0 });
  ParameterGradientCheck(
      stack,
      training_x,
      weights,
      [&fc_layer] (const Matrix& p) -> void {
        fc_layer->weights_.value = p;
      },
      [fc_layer] () -> Matrix {
        return fc_layer->weights_.gradient;
      },
      0.001f,
      1);
}

TEST(FullyConnectedLayerTest, InputGradient) {
  std::shared_ptr<InMemoryDataSet> training = CreateTestCase1_TrainingData();
  Matrix training_x = training->GetBatchInput(0);
  Matrix training_y = training->GetBatchOutput(0);

  std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
  stack->AddLayer<FullyConnectedLayer>(2, 1);
  stack->AddLayer<BiasLayer>(1, false);
  stack->AddLayer<NonlinearityLayer>(activation_functions::Sigmoid());
  stack->AddLayer<L2ErrorLayer>();

  stack->GetLayer<L2ErrorLayer>(-1)->SetExpectedValue(training_y);
  stack->GetLayer<FullyConnectedLayer>(0)->weights_.value =
      Matrix(1, 2, 1,  { 4.2, -3.0 });

  InputGradientCheck(
      stack,
      training_x,
      0.001f,
      7);
}

