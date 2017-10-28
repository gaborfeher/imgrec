#include <iostream>
#include <memory>
#include <utility>

#include "apps/cifar10/cifar_data_set.h"
#include "infra/model.h"
#include "cnn/batch_normalization_layer.h"
#include "cnn/bias_layer.h"
#include "cnn/convolutional_layer.h"
#include "cnn/fully_connected_layer.h"
#include "cnn/input_image_normalization_layer.h"
#include "cnn/inverted_dropout_layer.h"
#include "cnn/layer_stack.h"
#include "cnn/nonlinearity_layer.h"
#include "cnn/pooling_layer.h"
#include "cnn/reshape_layer.h"
#include "cnn/softmax_error_layer.h"

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
  Model model(stack, 123, 1);
  model.Evaluate(*validation, &error, &accuracy);
  model.Train(*training, 5, 0.008, 0.008);
  model.Evaluate(*validation, &error, &accuracy);
}


void TrainConvolutionalModel() {
  std::shared_ptr<CifarDataSet> training = LoadTraining(400);
  std::shared_ptr<CifarDataSet> validation = LoadValidation(10);

  std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
  stack->AddLayer<InputImageNormalizationLayer>(32, 32, 3);
  // Convolutional layer #1:
  stack->AddLayer<ConvolutionalLayer>(30, 3, 3, 1, 3);
  stack->AddLayer<BatchNormalizationLayer>(30, true);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());
  // Convolutional layer #2:
  stack->AddLayer<ConvolutionalLayer>(30, 3, 3, 1, 30);
  stack->AddLayer<BatchNormalizationLayer>(30, true);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());

  stack->AddLayer<PoolingLayer>(2, 2);

  // Convolutional layer #3:
  stack->AddLayer<ConvolutionalLayer>(30, 5, 5, 2, 30);
  stack->AddLayer<BatchNormalizationLayer>(30, true);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());

  // Convolutional layer #4:
  stack->AddLayer<ConvolutionalLayer>(30, 5, 5, 2, 30);
  stack->AddLayer<BatchNormalizationLayer>(30, true);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());

  stack->AddLayer<PoolingLayer>(2, 2);

  // Convolutional layer #5:
  stack->AddLayer<ConvolutionalLayer>(30, 5, 5, 2, 30);
  stack->AddLayer<BatchNormalizationLayer>(30, true);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());

  // Convolutional layer #6:
  stack->AddLayer<ConvolutionalLayer>(30, 5, 5, 2, 30);
  stack->AddLayer<BatchNormalizationLayer>(30, true);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());

  stack->AddLayer<ReshapeLayer>(8, 8, 30);

  // Fully connected layer #1:
  stack->AddLayer<FullyConnectedLayer>(8 * 8 * 30, 50);
  stack->AddLayer<BatchNormalizationLayer>(50, false);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());
  // Fully connected layer #2:
  stack->AddLayer<FullyConnectedLayer>(50, 10);
  stack->AddLayer<BatchNormalizationLayer>(10, false);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());

  stack->AddLayer(std::make_shared<SoftmaxErrorLayer>());

  float error, accuracy;
  Model model(stack, 123, 2);
  // model.Evaluate(*validation, &error, &accuracy);
  model.Train(*training, 5, 0.0001, 0.00002);
  model.Evaluate(*validation, &error, &accuracy);
}

int main() {
  // TrainSingleLayerFCModel();
  // TrainTwoLayerFCModel();
  TrainConvolutionalModel();
  return 0;
}
