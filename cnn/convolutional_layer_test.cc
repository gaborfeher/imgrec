//#include <functional>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"

#include "cnn/convolutional_layer.h"
#include "cnn/error_layer.h"
#include "cnn/fully_connected_layer.h"
#include "cnn/layer_stack.h"
#include "cnn/layer_test_base.h"
#include "cnn/reshape_layer.h"
#include "cnn/model.h"
#include "cnn/sigmoid_layer.h"
#include "linalg/device_matrix.h"


/*
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
  DeviceMatrix a_grad = conv_layer->filters_gradients_;

  // Approximate gradient the numerical way:
  DeviceMatrix n_grad = ComputeNumericGradients(
      filters,
      [&conv_layer, &stack, training_x, error_layer] (const DeviceMatrix& x) -> float {
        conv_layer->filters_ = x;
        stack->Forward(training_x);
        return error_layer->GetError();
      });

  a_grad.Print();
  n_grad.Print();

  ExpectMatrixEquals(a_grad, n_grad, 0.01, 5);
}
*/

class ConvolutionalLayerGradientTest : public ::testing::Test {
 protected:
  void SimpleConvolutionWeightGradientTest(
      std::shared_ptr<LayerStack> stack,
      std::shared_ptr<ConvolutionalLayer> conv_layer,
      std::shared_ptr<ErrorLayer> error_layer,
      const DeviceMatrix& training_x,
      const DeviceMatrix& filters) {

    conv_layer->filters_ = filters;
    // Compute gradient the analytical way:
    stack->Forward(training_x);
    stack->Backward(DeviceMatrix());
    DeviceMatrix a_grad = conv_layer->filters_gradients_;

    // Approximate gradient the numerical way:
    DeviceMatrix n_grad = ComputeNumericGradients(
        filters,
        [&conv_layer, &stack, training_x, error_layer] (const DeviceMatrix& x) -> float {
          conv_layer->filters_ = x;
          stack->Forward(training_x);
          return error_layer->GetError();
        });
    // a_grad.Print(); n_grad.Print();
    ExpectMatrixEquals(a_grad, n_grad, 0.05, 5);
  }

  void SimpleConvolutionInputGradientTest(
      std::shared_ptr<LayerStack> stack,
      std::shared_ptr<ConvolutionalLayer> conv_layer,
      std::shared_ptr<ErrorLayer> error_layer,
      const DeviceMatrix& training_x,
      const DeviceMatrix& filters) {

    conv_layer->filters_ = filters;
    // Compute gradient the analytical way:
    stack->Forward(training_x);
    stack->Backward(DeviceMatrix());
    DeviceMatrix a_grad = conv_layer->input_gradients_;

    // Approximate gradient the numerical way:
    DeviceMatrix n_grad = ComputeNumericGradients(
        training_x,
        [&stack, error_layer] (const DeviceMatrix& x) -> float {
          stack->Forward(x);
          return error_layer->GetError();
        });
    // a_grad.Print(); n_grad.Print();
    ExpectMatrixEquals(a_grad, n_grad, 0.05, 5);
  }

  void SimpleConvolutionGradientTest(
      const DeviceMatrix& training_x,
      const DeviceMatrix& training_y,
      const DeviceMatrix& filters,
      std::shared_ptr<ConvolutionalLayer> conv_layer) {
    std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
    stack->AddLayer(conv_layer);
    std::shared_ptr<ErrorLayer> error_layer =
        std::make_shared<ErrorLayer>();
    stack->AddLayer(error_layer);
    error_layer->SetExpectedValue(training_y);

    SimpleConvolutionWeightGradientTest(
        stack, conv_layer, error_layer, training_x, filters);
    SimpleConvolutionInputGradientTest(
        stack, conv_layer, error_layer, training_x, filters);
  }
};

TEST_F(ConvolutionalLayerGradientTest, Gradient1) {
  DeviceMatrix training_x(3, 3, 1, (float[]) {
      -1, 1, -2,
      2, -0.5, 0,
      -3, 2, 0
  });
  DeviceMatrix training_y(1, 1, 1, (float[]) {
      42.0
  });
  DeviceMatrix filters(3, 3, 1, (float[]) {
      3, -2, 1,
      0, -0.5, 0.5,
      -1, 0.5, 0,
  });

  SimpleConvolutionGradientTest(
      training_x,
      training_y,
      filters,
      std::make_shared<ConvolutionalLayer>(
          1, 3, 3,
          0, 1, 1));
}

TEST_F(ConvolutionalLayerGradientTest, Gradient2) {
  DeviceMatrix training_x(3, 3, 1, (float[]) {
      -1, 1, -2,
      2, -0.5, 0,
      -3, 2, 0
  });
  DeviceMatrix training_y(2, 2, 1, (float[]) {
      42.0, -43.0,
      44.0, 45.0,
  });
  DeviceMatrix filters(2, 2, 1, (float[]) {
      3, -2,
      0, -0.5,
  });

  SimpleConvolutionGradientTest(
      training_x,
      training_y,
      filters,
      std::make_shared<ConvolutionalLayer>(
          1, 2, 2,
          0, 1, 1));
}

TEST_F(ConvolutionalLayerGradientTest, Gradient3) {
  DeviceMatrix training_x(4, 5, 1, (float[]) {
      -1,  1,  -2,  1,  -0.5,
       2, -0.5, 0, -1,  -2,
      -3,  2,   0, -1,   2,
       0, -2,   3,  1,  -0.5,
  });
  DeviceMatrix training_y(3, 3, 1, (float[]) {
       42.0, -43.0, 21.0,
       44.0,  45.0, 22.0,
      -14.0,  32.0, 27.0
  });
  DeviceMatrix filters(2, 3, 1, (float[]) {
      3, -2  , -1,
      0, -0.5,  1.5
  });

  SimpleConvolutionGradientTest(
      training_x,
      training_y,
      filters,
      std::make_shared<ConvolutionalLayer>(
          1, 3, 2,
          0, 1, 1));
}

TEST_F(ConvolutionalLayerGradientTest, Gradient_TwoLayers) {
  DeviceMatrix training_x(3, 3, 2, (float[]) {
      -1, 1, -2,  // Layer1
      2, -0.5, 0,
      -3, 2, 0,
      2, 3, -1,  // Layer2
      0, 1, -2,
      -3, 0, 1,
  });
  DeviceMatrix training_y(1, 1, 1, (float[]) {
      42.0
  });
  DeviceMatrix filters(3, 3, 2, (float[]) {
      3, -2, 1,  // Layer1
      0, -0.5, 0.5,
      -1, 0.5, 0,
      0.5, 0.5, -2,  // Layer2
      -1, 3, 0,
      0, -2, 1
  });

  SimpleConvolutionGradientTest(
      training_x,
      training_y,
      filters,
      std::make_shared<ConvolutionalLayer>(
          1, 3, 3,
          0, 2, 1));
}

TEST_F(ConvolutionalLayerGradientTest, Gradient_TwoImagesXTwoLayers) {
  DeviceMatrix training_x(3, 3, 2 * 2, (float[]) {
      -1, 1, -2,  // Img1. Layer1
      2, -0.5, 0,
      -3, 2, 0,
      2, 3, -1,  // Img1. Layer2
      0, 1, -2,
      -3, 0, 1,
      0, -2, -3,  // Img2. Layer1
      -0.5, 1, 2,
      3, 0, 0,
      -2, 1, 0,  // Img2. Layer2
      -1, 0, -1,
      0, -3, 0,
  });
  DeviceMatrix training_y(1, 1, 2, (float[]) {
      42.0,
      -42.0,
  });
  DeviceMatrix filters(3, 3, 2, (float[]) {
      3, -2, 1,  // Filter Layer1
      0, -0.5, 0.5,
      -1, 0.5, 0,
      0.5, 0.5, -2,  // Filter Layer2
      -1, 3, 0,
      0, -2, 1
  });

  SimpleConvolutionGradientTest(
      training_x,
      training_y,
      filters,
      std::make_shared<ConvolutionalLayer>(
          1, 3, 3,
          0, 2, 1));
}

TEST_F(ConvolutionalLayerGradientTest, Gradient_TwoImagesXThreeLayers_Big) {
  DeviceMatrix training_x(4, 5, 6, (float[]) {
      -1,  1, -2,  1,  0,  // Img1. layer 1
       2, -0,  0, -1, -2,
      -3,  2,  0, -1,  2,
       0, -2,  3,  1,  0,
       1, -1,  0,  2,  0,  // Img1. layer 2
       1,  0,  1, -2,  1,
       1, -1,  2,  0,  0,
       0, -2,  3,  0,  1,
       0,  0, -5,  1, -1,  // Img1. layer 3
       2, -3,  0,  1, -1,
      -1, -1, -3,  0, -2,
      -2,  0, -2,  0,  2,

       1, -1,  0,  2,  0,  // Img2. layer 1
       1,  0,  1, -2,  1,
       1, -1,  2,  0,  0,
       0, -2,  3,  0,  1,
       0,  0, -2,  1, -1,  // Img2. layer 2
       2, -3,  0,  1, -1,
      -1, -1, -3,  0, -2,
      -2,  0, -2,  0,  2,
      -1,  1, -2,  1,  0,  // Img2. layer 3
       2, -0,  0, -1, -2,
      -3,  2,  0, -1,  2,
       0, -2,  3,  1,  0,
  });
  DeviceMatrix training_y(3, 3, 2, (float[]) {
        2.0,  -3.0,  1.0,  // Img1. output
        4.0,   5.0,  2.0,
       -4.0,   2.0,  7.0,
        8.0,   3.0,  7.0,  // Img2. output
        4.0,  -5.0,  1.0,
        8.0,   9.0,  2.0
  });
  DeviceMatrix filters(2, 3, 3, (float[]) {
      3, -2, -1,  // Filter Layer1
      1, -1,  1,
      2, -1, -2, // Filter Layer2
      0,  1,  0,
      1,  2, -1,  // Filter Layer3
     -2,  1, -2,
  });

  SimpleConvolutionGradientTest(
      training_x,
      training_y,
      filters,
      std::make_shared<ConvolutionalLayer>(
          1, 3, 2,
          0, 3, 1));
}

TEST_F(ConvolutionalLayerGradientTest, Gradient_TwoImagesXThreeLayersXTwoFilters_Big) {
  DeviceMatrix training_x(4, 5, 6, (float[]) {
      -1,  1, -2,  1,  0,  // Img1. layer 1
       2, -0,  0, -1, -2,
      -3,  2,  0, -1,  2,
       0, -2,  3,  1,  0,
       1, -1,  0,  2,  0,  // Img1. layer 2
       1,  0,  1, -2,  1,
       1, -1,  2,  0,  0,
       0, -2,  3,  0,  1,
       0,  0, -5,  1, -1,  // Img1. layer 3
       2, -3,  0,  1, -1,
      -1, -1, -3,  0, -2,
      -2,  0, -2,  0,  2,

       1, -1,  0,  2,  0,  // Img2. layer 1
       1,  0,  1, -2,  1,
       1, -1,  2,  0,  0,
       0, -2,  3,  0,  1,
       0,  0, -2,  1, -1,  // Img2. layer 2
       2, -3,  0,  1, -1,
      -1, -1, -3,  0, -2,
      -2,  0, -2,  0,  2,
      -1,  1, -2,  1,  0,  // Img2. layer 3
       2, -0,  0, -1, -2,
      -3,  2,  0, -1,  2,
       0, -2,  3,  1,  0,
  });
  DeviceMatrix training_y(3, 3, 4, (float[]) {
        2.0,  -3.0,  1.0,  // Img1. filter1. output
        4.0,   5.0,  2.0,
       -4.0,   2.0,  7.0,
       -3.0,   1.0,  2.0, // Img1. filter2. output
        5.0,   2.0,  4.0,
        2.0,   7.0, -4.0,
        8.0,   3.0,  7.0,  // Img2. filter1. output
        4.0,  -5.0,  1.0,
        8.0,   9.0,  2.0,
        3.0,   7.0,  8.0, // Img2. filter2. output
       -5.0,   1.0,  4.0,
        9.0,   2.0,  8.0,
  });
  DeviceMatrix filters(2, 3, 6, (float[]) {
      3, -2, -1,  // Filter1 Layer1
      1, -1,  1,
      2, -1, -2,  // Filter1 Layer2
      0,  1,  0,
      1,  2, -1,  // Filter1 Layer3
     -2,  1, -2,
     -1,  3, -2,  // Filter2 Layer1
      1,  1, -1,
     -2,  2, -1,  // Filter2 Layer2
      0,  0,  1,
     -1,  1,  2,  // Filter2 Layer3
     -2, -2,  1,
  });

  SimpleConvolutionGradientTest(
      training_x,
      training_y,
      filters,
      std::make_shared<ConvolutionalLayer>(
          2, 3, 2,
          0, 3, 1));
}

TEST_F(ConvolutionalLayerGradientTest, Gradient_FourImagesXTwoLayersXThreeFilters) {
  DeviceMatrix training_x(2, 2, 2 * 4, (float[]) {
      -1, 1,  // Img1. Layer1
      2, -1,
      2, 3,  // Img1. Layer2
      0, 1,
      0, -2,  // Img2. Layer1
      -0.5, 1,
      -2, 1,  // Img2. Layer2
      -1, 0,
      -3, 2, // Img3. Layer1
      -3, 0,
      3, 0,  // Img3. Layer2
      0, -3,
      -2, 0,  // Img4. Layer1
      -1, -2,
      -3, 2,  // Img4. Layer2
      0, -1,

  });
  DeviceMatrix training_y(1, 1, 12, (float[]) {
    3,
    1,
    -1,
    1,
    2,
    1,
    -3,
    1,
    -1,
    1,
    2,
    1,
  });
  DeviceMatrix filters(2, 2, 6, (float[]) {
      2, 0,  // Filter1 Layer1
      0, -2,
      1, -1,  // Filter1 Layer2
      -1, 1,
      1, 0,  // Filter2 Layer1
      0, -1,
      0, 1,  // Filter2 Layer2
      -1, 1,
      0, -2,  // Filter3 Layer1
      -2, 0,
      1, -1,  // Filter3 Layer2
      0, -1,
  });

  SimpleConvolutionGradientTest(
      training_x,
      training_y,
      filters,
      std::make_shared<ConvolutionalLayer>(
          3, 2, 2,
          0, 2, 1));
}


/*
TEST(LearnTest, StackInputGradientForConvolutionalTest) {
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
  conv_layer->filters_ = filters;
  conv_layer->Forward(training_x);
  DeviceMatrix s1_input = conv_layer->output();
  // Data preparation done.

  std::shared_ptr<SigmoidLayer> s1 =
      std::make_shared<SigmoidLayer>();
  stack->AddLayer(s1);
  stack->AddLayer(std::make_shared<ReshapeLayer>(1, 4, 2));
  stack->AddLayer(std::make_shared<FullyConnectedLayer>(8, 2));
  stack->AddLayer(std::make_shared<SigmoidLayer>());
  std::shared_ptr<ErrorLayer> error_layer =
      std::make_shared<ErrorLayer>();
  stack->AddLayer(error_layer);
  error_layer->SetExpectedValue(training_y);
//  error_layer->SetExpectedValue(DeviceMatrix(2, 8, 1));

  // Compute gradient the analytical way:
  stack->Forward(s1_input);
  stack->Backward(DeviceMatrix());
  s1->input_gradients().Print();
  DeviceMatrix a_grad = s1->input_gradients();

  // Approximate gradient the numerical way:
  DeviceMatrix n_grad = ComputeNumericGradients(
      s1_input,
      [&stack, error_layer] (const DeviceMatrix& x) -> float {
        stack->Forward(x);
        return error_layer->GetError();
      });
  std::cout << "NUM GRADS:" << std::endl;
  n_grad.Print();
  ExpectMatrixEquals(a_grad, n_grad, 0.01, 5);

}
*/
