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
#include "infra/trainer.h"
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
  stack->Initialize(std::make_shared<Random>(123));
  Trainer trainer(
      stack,
      std::make_shared<Logger>(2, "apps/cifar10/results/fc1"));
  trainer.Evaluate(*validation, &error, &accuracy);
  trainer.Train(*training, 10, GradientInfo(0.4, 0.01, GradientInfo::SGD), validation.get());
  trainer.Evaluate(*validation, &error, &accuracy);
}

void TrainTwoLayerFCModel(bool dropout, bool one_more_layer) {
  std::shared_ptr<CifarDataSet> training = LoadTraining(400);
  std::shared_ptr<CifarDataSet> validation = LoadValidation(10);

  std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
  stack->AddLayer<InputImageNormalizationLayer>(32, 32, 3);
  stack->AddLayer<ReshapeLayer>(32, 32, 3);

  stack->AddLayer<FullyConnectedLayer>(32 * 32 * 3, 128);
  stack->AddLayer<BatchNormalizationLayer>(128, false);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());

  if (one_more_layer) {
    stack->AddLayer<FullyConnectedLayer>(128, 128);
    stack->AddLayer<BatchNormalizationLayer>(128, false);
    stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());
  }

  if (dropout) {
    std::shared_ptr<Random> rnd = std::make_shared<Random>(123456);
    stack->AddLayer<InvertedDropoutLayer>(128, false, 0.8, rnd);
  }

  stack->AddLayer<FullyConnectedLayer>(128, 10);
  stack->AddLayer<BiasLayer>(10, false);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());

  stack->AddLayer(std::make_shared<SoftmaxErrorLayer>());

  std::string log_dir_name = "apps/cifar10/results/fc";
  if (one_more_layer) {
    log_dir_name += "3";
  } else {
    log_dir_name += "2";
  }
  if (dropout) {
    log_dir_name += "drop";
  } else {
    log_dir_name += "nodrop";
  }
  stack->Initialize(std::make_shared<Random>(123));
  Trainer trainer(
      stack,
      std::make_shared<Logger>(2, log_dir_name));
  float error, accuracy;
  trainer.Evaluate(*validation, &error, &accuracy);
  trainer.Train(
      *training,
      10,
      GradientInfo(0.006, 0.002, GradientInfo::ADAM),
      validation.get() /* evaluate after each epoch */);
}

void TrainConvolutional_1_Model() {
  std::shared_ptr<Random> rnd = std::make_shared<Random>(123456);
  float dropout = 0.5f;

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
  // Convolutional layer #3:
  stack->AddLayer<ConvolutionalLayer>(8, 3, 3, 1, 8);
  stack->AddLayer<BatchNormalizationLayer>(8, true);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());
  // Convolutional layer #4:
  stack->AddLayer<ConvolutionalLayer>(24, 5, 5, 2, 8);
  stack->AddLayer<BatchNormalizationLayer>(24, true);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());
  // Convolutional layer #5:
  stack->AddLayer<ConvolutionalLayer>(24, 5, 5, 2, 24);
  stack->AddLayer<BatchNormalizationLayer>(24, true);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());

  stack->AddLayer<PoolingLayer>(2, 2);

  // Convolutional layer #6:
  stack->AddLayer<ConvolutionalLayer>(24, 5, 5, 2, 24);
  stack->AddLayer<BatchNormalizationLayer>(24, true);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());

  stack->AddLayer<PoolingLayer>(2, 2);

  // Convolutional layer #7:
  stack->AddLayer<ConvolutionalLayer>(24, 5, 5, 2, 24);
  stack->AddLayer<BatchNormalizationLayer>(24, true);
  stack->AddLayer<InvertedDropoutLayer>(24, true, dropout, rnd);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());

  stack->AddLayer<ReshapeLayer>(8, 8, 24);

  // Fully connected layer #1:
  stack->AddLayer<FullyConnectedLayer>(8 * 8 * 24, 10);
  stack->AddLayer<BatchNormalizationLayer>(10, false);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());

  stack->AddLayer(std::make_shared<SoftmaxErrorLayer>());

  stack->Initialize(std::make_shared<Random>(123));
  Trainer trainer(
      stack,
      std::make_shared<Logger>(
          2,
          "apps/cifar10/results/conv1"));
  // model.Evaluate(*validation, &error, &accuracy);
  trainer.Train(
      *training,
      20,  // number of training epochs
      GradientInfo(
          0.0006,   // learning rate
          0.00012,  // L2 regularization
          GradientInfo::ADAM),  // optimization algorithm
      validation.get());  // evaluate on validation set after each epoch
  // model.Evaluate(*validation, &error, &accuracy);
}

void TrainConvolutional_2_Model() {
  std::shared_ptr<Random> rnd = std::make_shared<Random>(123456);
  float dropout = 0.5f;

  std::shared_ptr<CifarDataSet> training = LoadTraining(400);
  std::shared_ptr<CifarDataSet> validation = LoadValidation(10);

  std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
  stack->AddLayer<InputImageNormalizationLayer>(32, 32, 3);
  // Convolutional layer #1:
  stack->AddLayer<ConvolutionalLayer>(16, 3, 3, 1, 3);
  stack->AddLayer<BatchNormalizationLayer>(16, true);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());
  // Convolutional layer #2:
  stack->AddLayer<ConvolutionalLayer>(16, 3, 3, 1, 16);
  stack->AddLayer<BatchNormalizationLayer>(16, true);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());
  // Convolutional layer #3:
  stack->AddLayer<ConvolutionalLayer>(16, 3, 3, 1, 16);
  stack->AddLayer<BatchNormalizationLayer>(16, true);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());
  // Convolutional layer #4:
  stack->AddLayer<ConvolutionalLayer>(32, 5, 5, 2, 16);
  stack->AddLayer<BatchNormalizationLayer>(32, true);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());
  // Convolutional layer #5:
  stack->AddLayer<ConvolutionalLayer>(32, 5, 5, 2, 32);
  stack->AddLayer<BatchNormalizationLayer>(32, true);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());

  stack->AddLayer<PoolingLayer>(2, 2);

  // Convolutional layer #6:
  stack->AddLayer<ConvolutionalLayer>(32, 5, 5, 2, 32);
  stack->AddLayer<BatchNormalizationLayer>(32, true);
  stack->AddLayer<InvertedDropoutLayer>(32, true, dropout, rnd);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());

  stack->AddLayer<PoolingLayer>(2, 2);

  // Convolutional layer #7:
  stack->AddLayer<ConvolutionalLayer>(32, 5, 5, 2, 32);
  stack->AddLayer<BatchNormalizationLayer>(32, true);
  stack->AddLayer<InvertedDropoutLayer>(32, true, dropout, rnd);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());

  stack->AddLayer<ReshapeLayer>(8, 8, 32);

  // Fully connected layer #1:
  stack->AddLayer<FullyConnectedLayer>(8 * 8 * 32, 10);
  stack->AddLayer<BatchNormalizationLayer>(10, false);
  stack->AddLayer<NonlinearityLayer>(::activation_functions::LReLU());

  stack->AddLayer(std::make_shared<SoftmaxErrorLayer>());

  stack->Initialize(std::make_shared<Random>(123));
  Trainer trainer(
      stack,
      std::make_shared<Logger>(
          2,
          "apps/cifar10/results/conv2"));
  trainer.Train(
      *training,
      20,  // number of training epochs
      GradientInfo(
          0.0006,   // learning rate
          0.00012,  // L2 regularization
          GradientInfo::ADAM),  // optimization algorithm
      validation.get());  // evaluate on validation set after each epoch
}

int main() {
  TrainSingleLayerFCModel();
  TrainTwoLayerFCModel(false, false);
  TrainTwoLayerFCModel(true, false);
  TrainTwoLayerFCModel(false, true);
  TrainTwoLayerFCModel(true, true);
  TrainConvolutional_1_Model();
  TrainConvolutional_2_Model();
  return 0;
}
