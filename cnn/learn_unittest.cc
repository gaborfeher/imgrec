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
  stack->AddLayer(std::make_shared<SigmoidLayer>());
  std::shared_ptr<ErrorLayer> error_layer =
      std::make_shared<ErrorLayer>();
  stack->AddLayer(error_layer);

  error_layer->SetExpectedValue(training_y);
  // Set the weights of the fully connected layer. We want to
  // verify gradients at this point.
  fc_layer->weights_ =
      DeviceMatrix(1, 3, 1, (float[]) { 4.2, -3.0, 1.7});

  // Compute gradient the analytical way:
  stack->Forward(training_x);
  float error0 = error_layer->GetError();
  DeviceMatrix dummy;
  stack->Backward(dummy);

  fc_layer->weights_.Print();
  fc_layer->weights_gradients_.Print();

  // Compute gradient the numerical way:
  float delta = 0.01f;

  fc_layer->weights_ =
      DeviceMatrix(1, 3, 1, (float[]) { 4.2f + delta, -3.0f, 1.7f});
  stack->Forward(training_x);
  float error1 = error_layer->GetError();
  
  fc_layer->weights_ =
      DeviceMatrix(1, 3, 1, (float[]) { 4.2f, -3.0f + delta, 1.7f});
  stack->Forward(training_x);
  float error2 = error_layer->GetError();

  fc_layer->weights_ =
      DeviceMatrix(1, 3, 1, (float[]) { 4.2f, -3.0f, 1.7f + delta});
  stack->Forward(training_x);
  float error3 = error_layer->GetError();

  float x = 3.96f;
  std::vector<float> gradient {
      (error1 - error0) / delta * x,
      (error2 - error0) / delta * x,
      (error3 - error0) / delta * x
  };

  std::cout << gradient[0] << " " << gradient[1] << " " << gradient[2] << std::endl;
}
