#include <functional>
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

TEST(LayerStackTest, SaveLoad) {
  // create just some small fake network
  // TODO: test other types of layers
  std::shared_ptr<LayerStack> stack1 = std::make_shared<LayerStack>();
  stack1->AddLayer(std::make_shared<ConvolutionalLayer>(
      2, 3, 3,
      0, 2));
  stack1->AddLayer<FullyConnectedLayer>(8, 2);
  stack1->AddLayer<FullyConnectedLayer>(2, 10);

  Random random(42);
  stack1->Initialize(&random);

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
  // TODO: compare other members
  // TODO: strict binary float equality for matrices
  ExpectMatrixEquals(
      stack1->GetLayer<ConvolutionalLayer>(0)->filters_.value,
      stack2->GetLayer<ConvolutionalLayer>(0)->filters_.value);
  ExpectMatrixEquals(
      stack1->GetLayer<FullyConnectedLayer>(1)->weights_.value,
      stack2->GetLayer<FullyConnectedLayer>(1)->weights_.value);
  ExpectMatrixEquals(
      stack1->GetLayer<FullyConnectedLayer>(2)->weights_.value,
      stack2->GetLayer<FullyConnectedLayer>(2)->weights_.value);
}

