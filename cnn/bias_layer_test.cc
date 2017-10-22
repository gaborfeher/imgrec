#include <iostream>
#include <memory>

#include "cnn/bias_layer.h"
#include "cnn/l2_error_layer.h"
#include "cnn/layer_stack.h"
#include "cnn/layer_test_base.h"
#include "linalg/device_matrix.h"
#include "linalg/matrix_test_util.h"
#include "util/random.h"

#include "gtest/gtest.h"

TEST(BiasLayerTest, GradientCheck_ColumnMode) {
  DeviceMatrix training_x(4, 3, 1, (float[]) {
    1, 2, 3,
    1, 1, 1,
    1, 1, 2,
    1, 2, 3,
  });
  DeviceMatrix training_y(4, 3, 1, (float[]) {
    2, 3, 4,
    -1, -1, -1,
    2, 2, 1,
    2, 4, 6,
  });

  std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
  stack->AddLayer(std::make_shared<BiasLayer>(4, false));
  stack->AddLayer(std::make_shared<L2ErrorLayer>());
  std::shared_ptr<BiasLayer> bias_layer = stack->GetLayer<BiasLayer>(0);
  std::shared_ptr<ErrorLayer> error_layer = stack->GetLayer<ErrorLayer>(1);
  error_layer->SetExpectedValue(training_y);

  Random random(42);
  stack->Initialize(&random);  // (Bias layer always inits to zero.)
  
  DeviceMatrix biases(4, 1, 1, (float[]) { 0, -1, 1, 2} );

  ParameterGradientCheck(
      stack,
      training_x,
      biases,
      [&bias_layer] (const DeviceMatrix& p) -> void {
          bias_layer->biases_ = p;
      },
      [bias_layer] () -> DeviceMatrix {
          return bias_layer->biases_gradient_;
      },
      0.001f,
      1.0f);
}

TEST(BiasLayerTest, GradientCheck_LayerMode) {
  DeviceMatrix training_x(2, 3, 2, (float[]) {
    1, 2, 3,
    1, 1, 1,

    2, 1, 2,
    3, 3, 1,
  });
  DeviceMatrix training_y(2, 3, 2, (float[]) {
    0, -1, -1,
    2, 1, 1,

    1, 2, 1,
    1, 1, 1,
  });

  std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
  stack->AddLayer(std::make_shared<BiasLayer>(2, true));
  stack->AddLayer(std::make_shared<L2ErrorLayer>());
  std::shared_ptr<BiasLayer> bias_layer = stack->GetLayer<BiasLayer>(0);
  std::shared_ptr<ErrorLayer> error_layer = stack->GetLayer<ErrorLayer>(1);
  error_layer->SetExpectedValue(training_y);

  Random random(42);
  stack->Initialize(&random);  // (Bias layer always inits to zero.)
  
  DeviceMatrix biases(1, 1, 2, (float[]) { -1, 1 } );

  ParameterGradientCheck(
      stack,
      training_x,
      biases,
      [&bias_layer] (const DeviceMatrix& p) -> void {
          bias_layer->biases_ = p;
      },
      [bias_layer] () -> DeviceMatrix {
          return bias_layer->biases_gradient_;
      },
      0.001f,
      1.0f);
}
