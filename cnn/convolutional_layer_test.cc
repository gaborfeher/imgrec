
#include <iostream>
#include <random>
#include <vector>

#include "cereal/archives/portable_binary.hpp"
#include "cereal/types/memory.hpp"
#include "gtest/gtest.h"

#include "cnn/batch_normalization_layer.h"
#include "cnn/bias_layer.h"
#include "cnn/convolutional_layer.h"
#include "cnn/fully_connected_layer.h"
#include "cnn/input_image_normalization_layer.h"
#include "cnn/l2_error_layer.h"
#include "cnn/layer_stack.h"
#include "cnn/layer_test_base.h"
#include "cnn/nonlinearity_layer.h"
#include "cnn/reshape_layer.h"
#include "cnn/softmax_error_layer.h"
#include "infra/data_set.h"
#include "infra/logger.h"
#include "infra/trainer.h"
#include "linalg/matrix.h"
#include "linalg/matrix_test_util.h"
#include "util/random.h"


class ConvolutionalLayerGradientTest : public ::testing::Test {
 protected:
  void SimpleConvolutionGradientTest(
      const Matrix& training_x,
      const Matrix& training_y,
      const Matrix& filters,
      std::shared_ptr<ConvolutionalLayer> conv_layer) {
    std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
    stack->AddLayer(conv_layer);
    std::shared_ptr<L2ErrorLayer> error_layer =
        std::make_shared<L2ErrorLayer>();
    stack->AddLayer(error_layer);
    error_layer->SetExpectedValue(training_y);

    {
      SCOPED_TRACE("filters gradient check");
      ParameterGradientCheck(
          stack,
          training_x,
          filters,
          [&conv_layer] (const Matrix& p) -> void {
              conv_layer->filters_.value = p;
          },
          [conv_layer] () -> Matrix {
              return conv_layer->filters_.gradient;
          },
          0.08f,
          0.5f);
    }

    conv_layer->filters_.value = filters;
    {
      SCOPED_TRACE("input gradient check");
      InputGradientCheck(
          stack,
          training_x,
          0.05f,
          1.0f);
    }
  }
};

TEST_F(ConvolutionalLayerGradientTest, Gradient1) {
  Matrix training_x(3, 3, 1, {
      -1, 1, -2,
      2, -0.5, 0,
      -3, 2, 0
  });
  Matrix training_y(1, 1, 1, {
      42.0
  });
  Matrix filters(3, 3, 1, {
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
          0, 1));
}

TEST_F(ConvolutionalLayerGradientTest, Gradient1_Padding) {
  Matrix training_x(3, 3, 1, {
      -1, 1, -2,
      2, -0.5, 0,
      -3, 2, 0
  });
  Matrix training_y(3, 3, 1, {
      4.2, -4.2, 4.2,
      -4.2, 4.2, -4.2,
      4.2, -4.2, 4.2,
  });
  Matrix filters(3, 3, 1, {
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
          1, 1));
}

TEST_F(ConvolutionalLayerGradientTest, Gradient2) {
  Matrix training_x(3, 3, 1, {
      -1, 1, -2,
      2, -0.5, 0,
      -3, 2, 0
  });
  Matrix training_y(2, 2, 1, {
      42.0, -43.0,
      44.0, 45.0,
  });
  Matrix filters(2, 2, 1, {
      3, -2,
      0, -0.5,
  });

  SimpleConvolutionGradientTest(
      training_x,
      training_y,
      filters,
      std::make_shared<ConvolutionalLayer>(
          1, 2, 2,
          0, 1));
}

TEST_F(ConvolutionalLayerGradientTest, Gradient3) {
  Matrix training_x(4, 5, 1, {
      -1,  1,  -2,  1,  -0.5,
       2, -0.5, 0, -1,  -2,
      -3,  2,   0, -1,   2,
       0, -2,   3,  1,  -0.5,
  });
  Matrix training_y(3, 3, 1, {
       42.0, -43.0, 21.0,
       44.0,  45.0, 22.0,
      -14.0,  32.0, 27.0
  });
  Matrix filters(2, 3, 1, {
      3, -2  , -1,
      0, -0.5,  1.5
  });

  SimpleConvolutionGradientTest(
      training_x,
      training_y,
      filters,
      std::make_shared<ConvolutionalLayer>(
          1, 3, 2,
          0, 1));
}

TEST_F(ConvolutionalLayerGradientTest, Gradient_TwoLayers) {
  Matrix training_x(3, 3, 2, {
      -1, 1, -2,  // Layer1
      2, -0.5, 0,
      -3, 2, 0,
      2, 3, -1,  // Layer2
      0, 1, -2,
      -3, 0, 1,
  });
  Matrix training_y(1, 1, 1, {
      42.0
  });
  Matrix filters(3, 3, 2, {
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
          0, 2));
}

TEST_F(ConvolutionalLayerGradientTest, Gradient_TwoImagesXTwoLayers) {
  Matrix training_x(3, 3, 2 * 2, {
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
  Matrix training_y(1, 1, 2, {
      42.0,
      -42.0,
  });
  Matrix filters(3, 3, 2, {
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
          0, 2));
}

TEST_F(ConvolutionalLayerGradientTest, Gradient_TwoImagesXThreeLayers_Big) {
  Matrix training_x(4, 5, 6, {
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
  Matrix training_y(3, 3, 2, {
        2.0,  -3.0,  1.0,  // Img1. output
        4.0,   5.0,  2.0,
       -4.0,   2.0,  7.0,
        8.0,   3.0,  7.0,  // Img2. output
        4.0,  -5.0,  1.0,
        8.0,   9.0,  2.0
  });
  Matrix filters(2, 3, 3, {
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
          0, 3));
}

TEST_F(ConvolutionalLayerGradientTest, Gradient_TwoImagesXThreeLayersXTwoFilters_Big) {
  Matrix training_x(4, 5, 6, {
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
  Matrix training_y(3, 3, 4, {
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
  Matrix filters(2, 3, 6, {
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
          0, 3));
}

TEST_F(ConvolutionalLayerGradientTest, Gradient_FourImagesXTwoLayersXThreeFilters) {
  Matrix training_x(2, 2, 2 * 4, {
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
  Matrix training_y(1, 1, 12, {
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
  Matrix filters(2, 2, 6, {
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
          0, 2));
}

// For convolutional
void CreateTestCase1(
    Matrix* training_x,
    Matrix* training_y) {

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
  *training_x = Matrix(3, 6, 8 * 2, {
      // Image 1: pattern1 on the left, slightly damaged pattern2 on the right
      1, 0, 0, 0, 1, 0,  // layer1
      0, 1, 0, 1, 0, 1,
      0, 0, 1, 0, 1, 0,
      0, 0, 1, 1, 1, 1,  // layer2
      0, 1, 0, 1, 1, 1,
      1, 0, 0, 1, 1, 1,

      // Image 2: pattern2 on the left, slightly damaged pattern1 on the right
      0, 1, 0, 1, 0, 0, // layer1
      1, 1, 1, 0, 1, 1,
      0, 1, 0, 0, 0, 1,
      1, 1, 1, 0, 0, 1, // layer2
      1, 0, 1, 0, 0, 0,
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
  // 1=pattern1, 2=pattern2, 0=junk
  *training_y = Matrix(1, 8, 1, {
      1, 2, 1, 2, 1, 2, 0, 0,
  });
}

void Copy3x3VectorBlock(
    const std::vector<float>& src,
    std::vector<float>* dst, int dst_pos) {
  // src is 3x3, dst is 6x3
  for (int layer = 0; layer < 2; ++layer) {
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        (*dst)[layer * 6 * 3 + row * 6 + col + dst_pos] = 2 * src[layer * 3 * 3 + row * 3 + col];
      }
    }
  }
}

std::shared_ptr<InMemoryDataSet> CreateTestCase2(
    int num_batches,
    int num_samples_per_batch,
    int random_seed) {
  std::shared_ptr<InMemoryDataSet> training_ds =
      std::make_shared<InMemoryDataSet>(num_samples_per_batch);

  // We want to teach the convolutional layer to detect two patterns.
  // (Same as CreateTestCase1, but with more data.)
  // Patern1 (layer1+layer2):
  // 100 001
  // 010 010
  // 001 100
  // Pattern2 (layer1+layer2):
  // 010 111
  // 111 101
  // 010 111
  //

  std::vector<float> sample1 {
      1, 0, 0,  // layer1
      0, 1, 0,
      0, 0, 1,
      0, 0, 1,  // layer2
      0, 1, 0,
      1, 0, 0,
  };

  std::vector<float> sample2 {
      0, 1, 0,  // layer1
      1, 1, 1,
      0, 1, 0,
      1, 1, 1,  // layer2
      1, 0, 1,
      1, 1, 1,
  };

  std::mt19937 rnd(random_seed);
  for (int i = 0; i < num_batches; ++i) {

    std::vector<float> x;
    std::vector<float> y;

    std::uniform_int_distribution<> dist01(0, 2);
    std::uniform_int_distribution<> dist3(0, 2);
    std::uniform_int_distribution<> dist4(0, 3);
    for (int i = 0; i < num_samples_per_batch; ++i) {
      std::vector<float> x0;
      for (int i = 0; i < 6 * 6; ++i) {
        x0.push_back(dist01(rnd));
      }
      std::vector<float> y0 { 0, 0 };

      int mode = dist3(rnd);
      if (mode == 1) {
        int left_pos = dist4(rnd);
        Copy3x3VectorBlock(sample1, &x0, left_pos);
      } else if (mode == 2) {
        int left_pos = dist4(rnd);
        Copy3x3VectorBlock(sample2, &x0, left_pos);
      }

      x.insert(x.end(), x0.begin(), x0.end());
      y.push_back(mode);
    }

    training_ds->AddBatch(
        Matrix(3, 6, num_samples_per_batch * 2, x),
        Matrix(1, num_samples_per_batch, 1, y));

  }

  return training_ds;
}


std::shared_ptr<LayerStack> CreateConvolutionalTestEnv(bool use_batch_normalization) {
  std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
  // A function to add either a BiasLayer or a BatchNormalizationLayer to the stack.
  // (These layers are interchangeable.)
  std::function< void (int num_neurons, bool layered) > add_bias_layer =
      [&stack, use_batch_normalization] (int num_neurons, bool layered) -> void {
        if (use_batch_normalization) {
          stack->AddLayer<BatchNormalizationLayer>(num_neurons, layered);
        } else {
          stack->AddLayer<BiasLayer>(num_neurons, layered);
        }
      };

  stack->AddLayer(std::make_shared<InputImageNormalizationLayer>(
      3, 6, 2));
  stack->AddLayer(std::make_shared<ConvolutionalLayer>(
      2, 3, 3,
      0, 2));
  add_bias_layer(2, true);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());
  stack->AddLayer<ReshapeLayer>(1, 4, 2);
  stack->AddLayer<FullyConnectedLayer>(8, 2);
  add_bias_layer(2, false);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());
  stack->AddLayer<FullyConnectedLayer>(2, 10);
  add_bias_layer(10, false);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());
  stack->AddLayer<FullyConnectedLayer>(10, 3);
  add_bias_layer(3, false);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());
  stack->AddLayer<SoftmaxErrorLayer>();
  return stack;
}

TEST(ConvolutionalLayerTest, IntegratedGradientTest) {
  Matrix training_x;
  Matrix training_y;
  CreateTestCase1(&training_x, &training_y);
  // TODO: test with bigger batches, investigate stability issues

  std::shared_ptr<LayerStack> stack = CreateConvolutionalTestEnv(false);
  Random random(44);
  stack->Initialize(std::make_shared<Random>(44));
  // Note: we will overwrite the above random initialization of
  // the convolutional layer, but we'll keep it for the fully
  // connected layer.
  std::shared_ptr<ConvolutionalLayer> conv_layer =
      stack->GetLayer<ConvolutionalLayer>(1);
  std::shared_ptr<BiasLayer> bias_layer =
      stack->GetLayer<BiasLayer>(2);
  std::shared_ptr<ErrorLayer> error_layer = stack->GetLayer<ErrorLayer>(-1);

  error_layer->SetExpectedValue(training_y);
  // We will check the gradient of filters at this point:
  Matrix filters = conv_layer->filters_.value;
  Matrix biases = bias_layer->biases_.value;

  {
    SCOPED_TRACE("gradient check on filters");
    ParameterGradientCheck(
        stack,
        training_x,
        filters,
        [&conv_layer] (const Matrix& p) -> void {
            conv_layer->filters_.value = p;
        },
        [conv_layer] () -> Matrix {
            return conv_layer->filters_.gradient;
        },
        0.003f,
        7.0f);
  }
  conv_layer->filters_.value = filters;

  // 2. Compute gradient on the biases:
  {
    SCOPED_TRACE("gradient check on biases");
    ParameterGradientCheck(
        stack,
        training_x,
        biases,
        [&bias_layer] (const Matrix& p) -> void {
            bias_layer->biases_.value = p;
        },
        [bias_layer] () -> Matrix {
            return bias_layer->biases_.gradient;
        },
        0.003f,
        2.5f);
  }
  bias_layer->biases_.value = biases;

  {
    SCOPED_TRACE("gradient check on inputs");
    InputGradientCheck(
        stack,
        training_x,
        0.001f,
        120.0f);  // Investigate
  }
}

TEST(ConvolutionalLayerTest, TrainTest_Small) {
  std::shared_ptr<InMemoryDataSet> training_ds = CreateTestCase2(1000, 20, 142);
  std::shared_ptr<InMemoryDataSet> test_ds = CreateTestCase2(10, 20, 143);
  std::shared_ptr<LayerStack> stack = CreateConvolutionalTestEnv(false);

  stack->Initialize(std::make_shared<Random>(43));
  Trainer trainer(
      stack,
      std::make_shared<Logger>(1));
  trainer.Train(
      *training_ds,
      5,  // epochs
      GradientInfo(
          0.03,  // learn_rate
          0.0002,  // regularization
          GradientInfo::ADAM),  // algorithm
      test_ds.get());

  float test_error;
  float test_accuracy;
  trainer.Evaluate(*test_ds, &test_error, &test_accuracy);
  EXPECT_LT(test_error, 0.01);
  EXPECT_FLOAT_EQ(1.0, test_accuracy);
  // stack->Print();
}

TEST(ConvolutionalLayerTest, TrainTest_SaveLoad) {
  std::shared_ptr<InMemoryDataSet> training_ds = CreateTestCase2(100, 20, 142);
  std::shared_ptr<InMemoryDataSet> test_ds = CreateTestCase2(10, 20, 143);
  std::shared_ptr<LayerStack> stack1 = CreateConvolutionalTestEnv(true);

  stack1->Initialize(std::make_shared<Random>(43));
  Trainer trainer1(stack1, std::make_shared<Logger>(1));
  // Train model a little to have non-trivial behavior. (Not
  // too much to avoid 100% accuracy.)
  trainer1.Train(
      *training_ds,
      1,  // epochs
      GradientInfo(
          0.001,  // learn_rate
          0.001,  // regularization
          GradientInfo::ADAM),  // algorithm
      test_ds.get());

  // Evaluate original model:
  float test_error1;
  float test_accuracy1;
  trainer1.Evaluate(*test_ds, &test_error1, &test_accuracy1);

  // Save+reload model:
  std::stringstream buffer;
  {
    cereal::PortableBinaryOutputArchive output(buffer);
    output(stack1);
  }
  std::shared_ptr<LayerStack> stack2;
  {
    cereal::PortableBinaryInputArchive input(buffer);
    input(stack2);
  }

  // Evaluate save+reloaded model (no initialization here):
  Trainer trainer2(stack2, std::make_shared<Logger>(1));
  float test_error2;
  float test_accuracy2;
  trainer2.Evaluate(*test_ds, &test_error2, &test_accuracy2);

  // Check that save+reloaded model behaves the same way as the
  // original.
  EXPECT_EQ(test_error1, test_error2);
  EXPECT_EQ(test_accuracy1, test_accuracy2);
}

TEST(ConvolutionalLayerTest, TrainTest_Big) {
  std::shared_ptr<InMemoryDataSet> training_ds = CreateTestCase2(500, 60, 142);
  std::shared_ptr<InMemoryDataSet> test_ds = CreateTestCase2(10, 20, 143);
  std::shared_ptr<LayerStack> stack = CreateConvolutionalTestEnv(false);

  stack->Initialize(std::make_shared<Random>(43));
  Trainer trainer(stack, std::make_shared<Logger>(1));
  trainer.Train(
      *training_ds,
      5,  // epochs
      GradientInfo(
          0.006,  // learn_rate
          0.001,  // regularization
          GradientInfo::ADAM),  // algorithm
      test_ds.get());

  float test_error;
  float test_accuracy;
  trainer.Evaluate(*test_ds, &test_error, &test_accuracy);
  EXPECT_LT(test_error, 0.01);
  EXPECT_FLOAT_EQ(1.0, test_accuracy);
  // stack->Print();
}

TEST(ConvolutionalLayerTest, TrainTest_BatchNorm_Big) {
  std::shared_ptr<InMemoryDataSet> training_ds = CreateTestCase2(500, 60, 142);
  std::shared_ptr<InMemoryDataSet> test_ds = CreateTestCase2(10, 20, 143);
  std::shared_ptr<LayerStack> stack = CreateConvolutionalTestEnv(true);

  stack->Initialize(std::make_shared<Random>(42));
  Trainer trainer(stack, std::make_shared<Logger>(1));
  trainer.Train(
      *training_ds,
      5,  // epochs
      GradientInfo(
          0.006,  // learn_rate
          0.0,  // regularization
          GradientInfo::ADAM),
      test_ds.get());

  float test_error;
  float test_accuracy;
  trainer.Evaluate(*test_ds, &test_error, &test_accuracy);
  // :-|
  EXPECT_LT(test_error, 0.1);
  EXPECT_LT(0.98, test_accuracy);
  // stack->Print();
}
