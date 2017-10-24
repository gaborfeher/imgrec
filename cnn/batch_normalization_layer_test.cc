#include "cnn/batch_normalization_layer.h"

#include <math.h>

#include "cnn/error_layer.h"
#include "cnn/l2_error_layer.h"
#include "cnn/layer_stack.h"
#include "cnn/layer_test_base.h"
#include "linalg/matrix_test_util.h"

#include "gtest/gtest.h"


TEST(BatchNormalizationLayerTest, ForwardNormalization_ColumnMode) {
  DeviceMatrix training_x(4, 3, 1, (float[]) {
    1, 2, 3,
    1, 1, 1,
    1, 1, 4,
   -2, 0, 2,
  });
  BatchNormalizationLayer batch_layer(4, false);
  batch_layer.Initialize(NULL);
  batch_layer.BeginPhase(Layer::TRAIN_PHASE, 0);
  batch_layer.Forward(training_x);

  {
    SCOPED_TRACE("mean_");
    ExpectMatrixEquals(
        DeviceMatrix(4, 1, 1, (float[]) {
            2, 1, 2, 0
        }),
        batch_layer.mean_);
  }
  {
    SCOPED_TRACE("shifted_");
    ExpectMatrixEquals(
        DeviceMatrix(4, 3, 1, (float[]) {
            -1, 0, 1,
            0, 0, 0,
            -1, -1, 2,
            -2, 0, 2,
        }),
        batch_layer.shifted_);
  }
  {
    SCOPED_TRACE("variance_");
    ExpectMatrixEquals(
        DeviceMatrix(4, 1, 1, (float[]) {
            2.0f / 3.0f,
            0.0f,
            2.0f,
            8.0f / 3.0f,
        }),
        batch_layer.variance_);
  }
  {
    SCOPED_TRACE("normalized_");
    float epsilon = batch_layer.epsilon_;
    ExpectMatrixEquals(
        DeviceMatrix(4, 3, 1, (float[]) {
            float(-1.0f / sqrt(2.0f / 3.0f + epsilon)),
            float(0.0f),
            float(1.0f / sqrt(2.0f / 3.0f + epsilon)),

            0.0f,
            0.0f,
            0.0f,

            float(-1.0f / sqrt(2.0f + epsilon)),
            float(-1.0f / sqrt(2.0f + epsilon)),
            float(2.0f / sqrt(2.0f + epsilon)),

            float(-2.0f / sqrt(8.0f / 3.0f + epsilon)),
            0.0f,
            float(2.0f / sqrt(8.0f / 3.0f + epsilon)),
        }),
        batch_layer.normalized_);
  }
}

TEST(BatchNormalizationLayerTest, ForwardBetaGamma_ColumnMode) {
  DeviceMatrix training_x(2, 2, 1, (float[]) {
      -1, 1,
      1, -1,
  });
  BatchNormalizationLayer batch_layer(2, false);
  batch_layer.beta_ = DeviceMatrix(2, 1, 1, (float[]) {
      1, 2,
  });
  batch_layer.gamma_ = DeviceMatrix(2, 1, 1, (float[]) {
      3, 4,
  });
  batch_layer.BeginPhase(Layer::TRAIN_PHASE, 0);
  batch_layer.Forward(training_x);

  float epsilon = batch_layer.epsilon_;
  {
    SCOPED_TRACE("normalized_");
    ExpectMatrixEquals(
        DeviceMatrix(2, 2, 1, (float[]) {
            -1, 1,
            1, -1,
        }),
        batch_layer.normalized_,
        epsilon, 0.1);
  }
  {
    SCOPED_TRACE("output_");
    ExpectMatrixEquals(
        DeviceMatrix(2, 2, 1, (float[]) {
            -3 + 1,  3 + 1,
             4 + 2, -4 + 2,
        }),
        batch_layer.output_,
        0.001, 0.1);
  }
}

TEST(BatchNormalizationLayerTest, Forward_LayerMode) {
  DeviceMatrix training_x(2, 3, 4, (float[]) {
    // img1 layer1
    1, 1, 1,
    1, 1, 1,
    // img1 layer2
    2, 2, 2,
    2, 2, 2,
    // img2 layer1
    3, 3, 3,
    3, 3, 3,
    // img2 layer2
    4, 4, 4,
    4, 4, 4,
  });
  BatchNormalizationLayer batch_layer(2, true);
  batch_layer.beta_ = DeviceMatrix(1, 1, 2, (float[]) {
      1, 2,
  });
  batch_layer.gamma_ = DeviceMatrix(1, 1, 2, (float[]) {
      3, 4,
  });
  batch_layer.BeginPhase(Layer::TRAIN_PHASE, 0);
  batch_layer.Forward(training_x);

  {
    SCOPED_TRACE("mean_");
    ExpectMatrixEquals(
        DeviceMatrix(1, 1, 2, (float[]) {
            2, 3,
        }),
        batch_layer.mean_);
  }
  {
    SCOPED_TRACE("shifted_");
    ExpectMatrixEquals(
        DeviceMatrix(2, 3, 4, (float[]) {
          // img1 layer1
          -1, -1, -1,
          -1, -1, -1,
          // img1 layer2
          -1, -1, -1,
          -1, -1, -1,
          // img2 layer1
          1, 1, 1,
          1, 1, 1,
          // img2 layer2
          1, 1, 1,
          1, 1, 1,
        }),
        batch_layer.shifted_);
  }
  {
    SCOPED_TRACE("variance_");
    ExpectMatrixEquals(
        DeviceMatrix(1, 1, 2, (float[]) {
            1, 1,
        }),
        batch_layer.variance_);
  }
  {
    SCOPED_TRACE("normalized_");
    float epsilon = batch_layer.epsilon_;
    ExpectMatrixEquals(
        DeviceMatrix(2, 3, 4, (float[]) {
          // img1 layer1
          -1, -1, -1,
          -1, -1, -1,
          // img1 layer2
          -1, -1, -1,
          -1, -1, -1,
          // img2 layer1
          1, 1, 1,
          1, 1, 1,
          // img2 layer2
          1, 1, 1,
          1, 1, 1,
        }),
        batch_layer.normalized_,
        epsilon, 0.1);
  }
  {
    SCOPED_TRACE("output_");
    ExpectMatrixEquals(
        DeviceMatrix(2, 3, 4, (float[]) {
          // img1 layer1
          -3 + 1, -3 + 1, -3 + 1,
          -3 + 1, -3 + 1, -3 + 1,
          // img1 layer2
          -4 + 2, -4 + 2, -4 + 2,
          -4 + 2, -4 + 2, -4 + 2,
          // img2 layer1
          3 + 1, 3 + 1, 3 + 1,
          3 + 1, 3 + 1, 3 + 1,
          // img2 layer2
          4 + 2, 4 + 2, 4 + 2,
          4 + 2, 4 + 2, 4 + 2,
        }),
        batch_layer.output_,
        0.001, 0.1);
  }
}

TEST(BatchNormalizationLayerTest, GradientCheck_ColumnMode) {
  DeviceMatrix training_x(4, 3, 1, (float[]) {
    1, 2, 3,
    1, 1, 1,
    1, 1, 2,
    1, 2, 3,
  });
  DeviceMatrix training_y(4, 3, 1, (float[]) {
    -100, 100, -100,
    100, -100, 100,
    -100, 100, -100,
    100, -100, 100,
  });

  std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
  stack->AddLayer(std::make_shared<BatchNormalizationLayer>(4, false));
  stack->AddLayer(std::make_shared<L2ErrorLayer>());
  std::shared_ptr<BatchNormalizationLayer> batch_layer = stack->GetLayer<BatchNormalizationLayer>(0);
  std::shared_ptr<ErrorLayer> error_layer = stack->GetLayer<ErrorLayer>(1);
  batch_layer->BeginPhase(Layer::TRAIN_PHASE, 0);
  error_layer->SetExpectedValue(training_y);

  Random random(42);
  stack->Initialize(&random);
  
  DeviceMatrix beta(4, 1, 1, (float[]) { 0, -1, 1, 2} );
  DeviceMatrix gamma(4, 1, 1, (float[]) { -1, -0.5, 3, 1.2} );
  batch_layer->beta_ = beta;
  batch_layer->gamma_ = gamma;

  {
    SCOPED_TRACE("beta");
    ParameterGradientCheck(
        stack,
        training_x,
        beta,
        [&batch_layer] (const DeviceMatrix& p) -> void {
            batch_layer->beta_ = p;
        },
        [batch_layer] () -> DeviceMatrix {
            return batch_layer->beta_gradient_;
        },
        0.01f,
        3.0f);
  }

  {
    SCOPED_TRACE("gamma");
    ParameterGradientCheck(
        stack,
        training_x,
        gamma,
        [&batch_layer] (const DeviceMatrix& p) -> void {
            batch_layer->gamma_ = p;
        },
        [batch_layer] () -> DeviceMatrix {
            return batch_layer->gamma_gradient_;
        },
        0.01f,
        60.0f);
  }

  {
    SCOPED_TRACE("input");
    InputGradientCheck(
        stack,
        training_x,
        0.03f,
        4.0f);
  }
}
