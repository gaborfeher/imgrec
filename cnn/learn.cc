#include <iostream>

#include "cnn/convolutional_layer.h"
#include "cnn/fully_connected_layer.h"
#include "cnn/error_layer.h"
#include "cnn/fully_connected_layer.h"
#include "cnn/layer_stack.h"
#include "cnn/model.h"
#include "cnn/sigmoid_layer.h"
#include "linalg/device_matrix.h"

int main() {
  DeviceMatrix training_x(
      DeviceMatrix(8, 3, (float[]) {
          -1,  2, 1,
           0,  1, 1,
           1,  0, 1,
           2, -1, 1,
          -2,  1, 1,
          -1,  0, 1,
           0, -1, 1,
           1, -2, 1,
        }).T());
  DeviceMatrix training_y(
      DeviceMatrix(8, 1, (float[]) {
          0,
          0,
          0,
          0,
          1,
          1,
          1,
          1,
      }).T());

  DeviceMatrix test_x(
      DeviceMatrix(2, 3, (float[]) {
          -1, -1, 1,
           1,  1, 1,
      }).T());
  DeviceMatrix test_y(
      DeviceMatrix(2, 1, (float[]) {
          1,
          0,
      }).T());

  std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
  stack->AddLayer(std::make_shared<ConvolutionalLayer>(
        5, 3, 3,
        1, 3, 1));
  stack->AddLayer(std::make_shared<SigmoidLayer>());
  Model model(
      stack,
      std::make_shared<ErrorLayer>());

  model.Train(training_x, training_y, 100, 5);
  model.Evaluate(test_x, test_y);

}
