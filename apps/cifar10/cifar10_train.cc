#include <iostream>
#include <memory>

#include "apps/cifar10/cifar_data_set.h"
#include "infra/model.h"
#include "cnn/bias_layer.h"
#include "cnn/fully_connected_layer.h"
#include "cnn/input_image_normalization_layer.h"
#include "cnn/layer_stack.h"
#include "cnn/nonlinearity_layer.h"
#include "cnn/softmax_error_layer.h"
#include "cnn/reshape_layer.h"

void TrainSingleLayerFCModel(const DataSet& training, const DataSet& validation) {
  std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
  stack->AddLayer(std::make_shared<InputImageNormalizationLayer>(32, 32, 3));
  stack->AddLayer(std::make_shared<ReshapeLayer>(32, 32, 3));
  stack->AddLayer(std::make_shared<FullyConnectedLayer>(32 * 32 * 3, 10));
  stack->AddLayer(std::make_shared<BiasLayer>(10, false));
  stack->AddLayer(std::make_shared<NonlinearityLayer>(::activation_functions::LReLU()));
  stack->AddLayer(std::make_shared<SoftmaxErrorLayer>());

  float error, accuracy;
  Model model(stack, 123, true);
  model.Evaluate(validation, &error, &accuracy);
  model.Train(training, 10, 0.4, 0.01);
  model.Evaluate(validation, &error, &accuracy);
}

void TrainTwoLayerFCModel(const DataSet& training, const DataSet& validation) {
  std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
  stack->AddLayer(std::make_shared<InputImageNormalizationLayer>(32, 32, 3));
  stack->AddLayer(std::make_shared<ReshapeLayer>(32, 32, 3));

  stack->AddLayer(std::make_shared<FullyConnectedLayer>(32 * 32 * 3, 50));
  stack->AddLayer(std::make_shared<BiasLayer>(50, false));
  stack->AddLayer(std::make_shared<NonlinearityLayer>(::activation_functions::LReLU()));

  stack->AddLayer(std::make_shared<FullyConnectedLayer>(50, 10));
  stack->AddLayer(std::make_shared<BiasLayer>(10, false));
  stack->AddLayer(std::make_shared<NonlinearityLayer>(::activation_functions::LReLU()));

  stack->AddLayer(std::make_shared<SoftmaxErrorLayer>());

  float error, accuracy;
  Model model(stack, 123, true);
  model.Evaluate(validation, &error, &accuracy);
  model.Train(training, 10, 0.3, 0.05);
  model.Evaluate(validation, &error, &accuracy);
}

int main() {
  CifarDataSet training(
    {
      "apps/cifar10/downloaded_deps/cifar-10-batches-bin/data_batch_1.bin",
      "apps/cifar10/downloaded_deps/cifar-10-batches-bin/data_batch_2.bin",
      "apps/cifar10/downloaded_deps/cifar-10-batches-bin/data_batch_3.bin",
      "apps/cifar10/downloaded_deps/cifar-10-batches-bin/data_batch_4.bin",
    },
    10);
  CifarDataSet validation(
    {
      "apps/cifar10/downloaded_deps/cifar-10-batches-bin/data_batch_5.bin",
    },
    10);

  TrainSingleLayerFCModel(training, validation);
  // TrainTwoLayerFCModel(training, validation);  // not working

  return 0;
}
