#include <iostream>
#include <memory>
#include <utility>

#include "apps/cifar10/cifar_data_set.h"
#include "infra/model.h"
#include "cnn/batch_normalization_layer.h"
#include "cnn/bias_layer.h"
#include "cnn/fully_connected_layer.h"
#include "cnn/input_image_normalization_layer.h"
#include "cnn/inverted_dropout_layer.h"
#include "cnn/layer_stack.h"
#include "cnn/nonlinearity_layer.h"
#include "cnn/softmax_error_layer.h"
#include "cnn/reshape_layer.h"

std::shared_ptr<CifarDataSet> LoadTraining(int minibatch_size) {
  return std::make_shared<CifarDataSet>(
      std::vector<std::string>
      {
        "apps/cifar10/downloaded_deps/cifar-10-batches-bin/data_batch_1.bin",
        "apps/cifar10/downloaded_deps/cifar-10-batches-bin/data_batch_2.bin",
        "apps/cifar10/downloaded_deps/cifar-10-batches-bin/data_batch_3.bin",
        "apps/cifar10/downloaded_deps/cifar-10-batches-bin/data_batch_4.bin",
      },
      minibatch_size);
}

std::shared_ptr<CifarDataSet> LoadValidation(int minibatch_size) {
  return std::make_shared<CifarDataSet>(
      std::vector<std::string>
      {
        "apps/cifar10/downloaded_deps/cifar-10-batches-bin/data_batch_5.bin",
      },
      minibatch_size);
}

void TrainSingleLayerFCModel() {
  std::shared_ptr<CifarDataSet> training = LoadTraining(10);
  std::shared_ptr<CifarDataSet> validation = LoadValidation(10);

  std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
  stack->AddLayer<InputImageNormalizationLayer>(32, 32, 3);
  stack->AddLayer<ReshapeLayer>(32, 32, 3);
  stack->AddLayer<FullyConnectedLayer>(32 * 32 * 3, 10);
  stack->AddLayer<BiasLayer>(10, false);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());
  stack->AddLayer<SoftmaxErrorLayer>();

  float error, accuracy;
  Model model(stack, 123, 1);
  model.Evaluate(*validation, &error, &accuracy);
  model.Train(*training, 5, 0.4, 0.01);
  model.Evaluate(*validation, &error, &accuracy);
}

void TrainTwoLayerFCModel() {
  std::shared_ptr<CifarDataSet> training = LoadTraining(400);
  std::shared_ptr<CifarDataSet> validation = LoadValidation(10);

  std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
  stack->AddLayer<InputImageNormalizationLayer>(32, 32, 3);
  stack->AddLayer<ReshapeLayer>(32, 32, 3);

  stack->AddLayer<FullyConnectedLayer>(32 * 32 * 3, 50);
  stack->AddLayer<BatchNormalizationLayer>(50, false);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());

  stack->AddLayer<FullyConnectedLayer>(50, 10);
  stack->AddLayer<BiasLayer>(10, false);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());

  stack->AddLayer(std::make_shared<SoftmaxErrorLayer>());

  float error, accuracy;
  Model model(stack, 123, true);
  model.Evaluate(*validation, &error, &accuracy);
  model.Train(*training, 5, 0.008, 0.008);
  model.Evaluate(*validation, &error, &accuracy);
}

int main() {
  TrainSingleLayerFCModel();
  TrainTwoLayerFCModel();

  return 0;
}
