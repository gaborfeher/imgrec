#include <iostream>
#include <memory>
#include <utility>

#include "apps/cifar10/cifar_data_set.h"
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
#include "infra/logger.h"
#include "infra/model.h"
#include "util/random.h"

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
  Model model(stack, 123, std::make_shared<Logger>(1));
  model.Evaluate(*validation, &error, &accuracy);
  model.Train(*training, 5, GradientInfo(0.4, 0.01, GradientInfo::SGD));
  model.Evaluate(*validation, &error, &accuracy);
}

void TrainTwoLayerFCModel(bool dropout) {
  std::shared_ptr<CifarDataSet> training = LoadTraining(400);
  std::shared_ptr<CifarDataSet> validation = LoadValidation(10);

  std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
  stack->AddLayer<InputImageNormalizationLayer>(32, 32, 3);
  stack->AddLayer<ReshapeLayer>(32, 32, 3);

  stack->AddLayer<FullyConnectedLayer>(32 * 32 * 3, 50);
  stack->AddLayer<BatchNormalizationLayer>(50, false);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());

  if (dropout) {
    std::shared_ptr<Random> rnd = std::make_shared<Random>(123456);
    stack->AddLayer<InvertedDropoutLayer>(50, false, 0.9, rnd);
  }

  stack->AddLayer<FullyConnectedLayer>(50, 10);
  stack->AddLayer<BiasLayer>(10, false);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());

  stack->AddLayer(std::make_shared<SoftmaxErrorLayer>());

  float error, accuracy;
  Model model(stack, 123, std::make_shared<Logger>(2));
  model.Evaluate(*validation, &error, &accuracy);
  model.Train(
      *training,
      5,
      GradientInfo(0.008, 0.008, GradientInfo::SGD),
      validation.get() /* evaluate after each epoch */);
  // model.Evaluate(*validation, &error, &accuracy);
}


void TrainConvolutionalModel() {
  std::shared_ptr<CifarDataSet> training = LoadTraining(400);
  std::shared_ptr<CifarDataSet> validation = LoadValidation(10);

  std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
  stack->AddLayer<InputImageNormalizationLayer>(32, 32, 3);
  // Convolutional layer #1:
  stack->AddLayer<ConvolutionalLayer>(8, 3, 3, 1, 3);
  stack->AddLayer<BatchNormalizationLayer>(8, true);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());
  // Convolutional layer #2:
  stack->AddLayer<ConvolutionalLayer>(8, 3, 3, 1, 8);
  stack->AddLayer<BatchNormalizationLayer>(8, true);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());

  stack->AddLayer<ConvolutionalLayer>(8, 3, 3, 1, 8);
  stack->AddLayer<BatchNormalizationLayer>(8, true);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());

  // Convolutional layer #3:
  stack->AddLayer<ConvolutionalLayer>(24, 5, 5, 2, 8);
  stack->AddLayer<BatchNormalizationLayer>(24, true);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());

  // Convolutional layer #4:
  stack->AddLayer<ConvolutionalLayer>(24, 5, 5, 2, 24);
  stack->AddLayer<BatchNormalizationLayer>(24, true);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());

  stack->AddLayer<PoolingLayer>(2, 2);

  // Convolutional layer #5:
  stack->AddLayer<ConvolutionalLayer>(24, 5, 5, 2, 24);
  stack->AddLayer<BatchNormalizationLayer>(24, true);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());

  stack->AddLayer<PoolingLayer>(2, 2);

  // Convolutional layer #6:
  stack->AddLayer<ConvolutionalLayer>(24, 5, 5, 2, 24);
  stack->AddLayer<BatchNormalizationLayer>(24, true);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());

  stack->AddLayer<ReshapeLayer>(8, 8, 24);

  // Fully connected layer #1:
  stack->AddLayer<FullyConnectedLayer>(8 * 8 * 24, 10);
  stack->AddLayer<BatchNormalizationLayer>(10, false);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());

  stack->AddLayer(std::make_shared<SoftmaxErrorLayer>());

  Model model(stack, 123, std::make_shared<Logger>(2));
  // model.Evaluate(*validation, &error, &accuracy);
  model.Train(
      *training,
      7,
      GradientInfo(0.0006, 0.00012, GradientInfo::ADAM),
      validation.get());
  // model.Evaluate(*validation, &error, &accuracy);
}

int main() {
  // TrainSingleLayerFCModel();
  // TrainTwoLayerFCModel(false);
  // TrainTwoLayerFCModel(true);
  TrainConvolutionalModel();
  return 0;
}
