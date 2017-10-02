#include <iostream>

#include "cnn/error_layer.h"
#include "cnn/fully_connected_layer.h"
#include "cnn/layer_stack.h"
#include "cnn/sigmoid_layer.h"
#include "linalg/device_matrix.h"

int main() {
  std::cout << "hello, world" << std::endl;

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

  LayerStack stack;
  stack.AddLayer(std::make_shared<FullyConnectedLayer>(3, 1));
  stack.AddLayer(std::make_shared<SigmoidLayer>());
  std::shared_ptr<ErrorLayer> error_layer = std::make_shared<ErrorLayer>();
  stack.AddLayer(error_layer);

  for (int i = 0; i < 100; ++i) {
    error_layer->SetExpectedValue(training_y);
    stack.Forward(training_x);
    stack.output().AssertDimensions(1, 1);
    std::cout << "Training Error= " << stack.output().GetVector()[0] << std::endl;
    DeviceMatrix dummy;
    stack.Backward(dummy);
    stack.ApplyGradient(10);
  }

  error_layer->SetExpectedValue(test_y);
  stack.Forward(test_x);
  stack.output().AssertDimensions(1, 1);
  std::cout << "Test Error= " << stack.output().GetVector()[0] << std::endl;
}
