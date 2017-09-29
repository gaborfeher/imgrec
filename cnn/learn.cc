#include <stdio>

#include "cnn/error_layer.h"
#include "cnn/fully_connected_layer.h"
#include "cnn/layer_stack.h"
#include "cnn/sigmoid_layer.h"
#include "linalg/device_matrix.h"
#include "linalg/host_matrix.h"

int main() {
  std::cout << "hello, world";

  HostMatrix training_x_host(8, 2, (float[]) {
      -1, 2,
      0, 1,
      1, 0,
      2, -1,
      -2, 1,
      -1, 0,
      0, -1,
      1, -2,
    });
  DeviceMatrix training_x(DeviceMatrix(training_x_host).T());
  HostMatrix training_y_host(8, 1, (float[]) {
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
    });
  DeviceMatrix training_y(DeviceMatrix(training_y_host).T());

  HostMatrix test_x_host(2, 2, (float[]) {
      -1, -1,
      1, 1,
    });
  DeviceMatrix test_x(DeviceMatrix(test_x_host).T());
  HostMatrix test_y_host(2, 1, (float[]) {
      1,
      0,
    });
  DeviceMatrix test_y(DeviceMatrix(test_y_host).T());

  LayerStack stack;
  stack.AddLayer(new FullyConnectedLayer(2, 1));
  stack.AddLayer(new SigmoidLayer);
  stack.AddLayer(new ErrorLayer(trainging_y_host));

  for (int i = 0; i < 100; ++i) {
    stack.Forward(training_x);
    HostMatrix(stack.output()).Dump();
    DeviceMatrix dummy;
    stack.Backward(dummy);
    stack.ApplyGradient(0.1);
  }

  stack.Forward(test_x);
  HostMatrix(stack.output()).Dump();
}
