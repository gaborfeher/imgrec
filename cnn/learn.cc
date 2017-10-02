#include <iostream>

#include "cnn/error_layer.h"
#include "cnn/fully_connected_layer.h"
#include "cnn/layer_stack.h"
#include "cnn/sigmoid_layer.h"
#include "linalg/device_matrix.h"

int main() {
  std::cout << "hello, world" << std::endl;

  DeviceMatrix training_x(
      DeviceMatrix(8, 2, (float[]) {
          -1, 2,
          0, 1,
          1, 0,
          2, -1,
          -2, 1,
          -1, 0,
          0, -1,
          1, -2,
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
      DeviceMatrix(2, 2, (float[]) {
          -1, -1,
          1, 1,
      }).T());
  DeviceMatrix test_y(
      DeviceMatrix(2, 1, (float[]) {
          1,
          0,
      }).T());

  LayerStack stack;
  stack.AddLayer(std::make_shared<FullyConnectedLayer>(2, 1));
  stack.AddLayer(std::make_shared<SigmoidLayer>());
  stack.AddLayer(std::make_shared<ErrorLayer>(training_y));

  for (int i = 0; i < 100; ++i) {
    stack.Forward(training_x);
    stack.output().Print();
    // TODO: clean up this dummy business (understand backprop better)
    DeviceMatrix dummy;
    stack.Backward(dummy);
    stack.ApplyGradient(0.1);
  }

  stack.Forward(test_x);
  stack.output().Print();
}