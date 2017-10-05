#include <functional>
#include <iostream>
#include <vector>

#include "cnn/convolutional_layer.h"
#include "cnn/error_layer.h"
#include "cnn/fully_connected_layer.h"
#include "cnn/layer_stack.h"
#include "cnn/reshape_layer.h"
#include "cnn/model.h"
#include "cnn/sigmoid_layer.h"
#include "linalg/device_matrix.h"

#include "gtest/gtest.h"

// For fully connected
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

// For convolutional
void CreateTestCase2(
    DeviceMatrix* training_x,
    DeviceMatrix* training_y) {

  // We want to teach the convolutional layer to detect two patterns:
  // Patern1 (layer1+layer2):
  // 100 001
  // 010 010
  // 001 100
  // Pattern2 (layer1+layer2):
  // 010 111
  // 111 101
  // 010 111
  //
  *training_x = DeviceMatrix(3, 6, 8 * 2, (float[]) {
      // Image 1: pattern1 and pattern2 side by side
      1, 0, 0, 0, 1, 0,  // layer1
      0, 1, 0, 1, 1, 1,
      0, 0, 1, 0, 1, 0,
      0, 0, 1, 1, 1, 1,  // layer2
      0, 1, 0, 1, 0, 1,
      1, 0, 0, 1, 1, 1,

      // Image 2: pattern2 and pattern1 side by side
      0, 1, 0, 1, 0, 0, // layer1
      1, 1, 1, 0, 1, 0,
      0, 1, 0, 0, 0, 1,
      1, 1, 1, 0, 0, 1, // layer2
      1, 0, 1, 0, 1, 0,
      1, 1, 1, 1, 0, 0,

      // Image 3: pattern1 starting at the 3rd column
      0, 0, 1, 0, 0, 0,  // layer1
      1, 1, 0, 1, 0, 0,
      0, 1, 0, 0, 1, 1,
      1, 1, 0, 0, 1, 0,  // layer2
      0, 0, 0, 1, 0, 0,
      1, 0, 1, 0, 0, 1,

      // Image 4: pattern2 starting at the 2nd column
      0, 0, 1, 0, 0, 0,  // layer1
      1, 1, 1, 1, 0, 0,
      0, 0, 1, 0, 1, 1,
      1, 1, 1, 1, 1, 0,  // layer2
      0, 1, 0, 1, 0, 0,
      1, 1, 1, 1, 0, 1,

      // Image 5: pattern1 starting at the 4th column
      0, 0, 1, 1, 0, 0,  // layer1
      1, 1, 1, 0, 1, 0,
      0, 0, 1, 0, 0, 1,
      1, 1, 1, 0, 0, 1,  // layer2
      0, 1, 0, 0, 1, 0,
      1, 1, 1, 1, 0, 0,

      // Image 6: pattern2 starting at the 1st column
      0, 1, 0, 0, 0, 0,  // layer1
      1, 1, 1, 1, 0, 0,
      0, 1, 0, 0, 1, 1,
      1, 1, 1, 1, 1, 0,  // layer2
      1, 0, 1, 1, 0, 0,
      1, 1, 1, 1, 0, 1,

      // Image 7: garbage
      0, 1, 0, 0, 0, 0,  // layer1
      1, 0, 0, 1, 0, 0,
      0, 1, 0, 0, 1, 1,
      1, 1, 1, 1, 1, 0,  // layer2
      1, 1, 0, 1, 0, 0,
      1, 1, 1, 1, 0, 1,

      // Image 8: garbage
      0, 0, 1, 0, 0, 0,  // layer1
      1, 1, 1, 1, 0, 0,
      1, 1, 1, 0, 1, 1,
      1, 1, 1, 1, 1, 1,  // layer2
      0, 0, 1, 0, 0, 0,
      1, 0, 1, 0, 0, 1,
  });
  *training_y =
      DeviceMatrix(8, 2, 1, (float[]) {
          1, 1,
          1, 1,
          1, 0,
          0, 1,
          1, 0,
          0, 1,
          0, 0,
          0, 0,
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

TEST(LearnTest, ConvolutionalGradient) {
  DeviceMatrix training_x;
  DeviceMatrix training_y;
  CreateTestCase2(&training_x, &training_y);

  // We will check the gradient of filters at this point:
  DeviceMatrix filters(3, 3, 4, (float[]) {
    1, -0.5, 1,
    0.5, 0.5, -1,
    -1, -1, -0.5,

    1, -0.5, 1,
    0.5, 0.5, -1,
    -1, -1, -0.5,
      
    1, -0.5, 1,
    0.5, 0.5, -1,
    -1, -1, -0.5,

    1, -0.5, 1,
    0.5, 0.5, -1,
    -1, -1, -0.5,
  });

  std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
  std::shared_ptr<ConvolutionalLayer> conv_layer =
      std::make_shared<ConvolutionalLayer>(
          2, 3, 3,
          0, 2, 1);
  stack->AddLayer(conv_layer);
  stack->AddLayer(std::make_shared<SigmoidLayer>());
  stack->AddLayer(std::make_shared<ReshapeLayer>(1, 4, 2));
  stack->AddLayer(std::make_shared<FullyConnectedLayer>(8, 2));
  stack->AddLayer(std::make_shared<SigmoidLayer>());
  std::shared_ptr<ErrorLayer> error_layer =
      std::make_shared<ErrorLayer>();
  stack->AddLayer(error_layer);
  error_layer->SetExpectedValue(training_y);
  conv_layer->filters_ = filters;

  // Compute gradient the analytical way:
  stack->Forward(training_x);
  stack->Backward(DeviceMatrix());
  conv_layer->filters_.Print();
  conv_layer->filters_gradients_.Print();
  std::cout << "ERROR= " << error_layer->GetError() << std::endl;
  std::cout << "ERROR= " << error_layer->GetError() << std::endl;

  // Approximate gradient the numerical way:
  DeviceMatrix n_grad_m = ComputeNumericGradients(
      filters,
      [&conv_layer, &stack, training_x, error_layer] (const DeviceMatrix& x) -> float {
        conv_layer->filters_ = x;
        stack->Forward(training_x);
        return error_layer->GetError();
      });
  std::cout << "NUM GRADS:" << std::endl;
  n_grad_m.Print();
}
