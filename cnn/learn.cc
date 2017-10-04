#include <iostream>

#include "cnn/convolutional_layer.h"
#include "cnn/fully_connected_layer.h"
#include "cnn/error_layer.h"
#include "cnn/fully_connected_layer.h"
#include "cnn/layer_stack.h"
#include "cnn/model.h"
#include "cnn/reshape_layer.h"
#include "cnn/sigmoid_layer.h"
#include "linalg/device_matrix.h"

int main() {
  // We want to teach the convolutional layer to detect two patterns:
  // Patern1 (layer1+layer2):
  // 100 001
  // 010 010
  // 001 100
  // Pattern2 (layer1+layer2):
  // 010 111
  // 111 101
  // 010 111
  //
  DeviceMatrix training_x(3, 6, 8 * 2, (float[]) {
      // Image 1: pattern1 and pattern2 side by side
      1, 0, 0, 0, 1, 0,  // layer1
      0, 1, 0, 1, 1, 1,
      0, 0, 1, 0, 1, 0,
      0, 0, 1, 1, 1, 1,  // layer2
      0, 1, 0, 1, 0, 1,
      1, 0, 0, 1, 1, 1,

      // Image 2: pattern2 and pattern1 side by side
      0, 1, 0, 1, 0, 0, // layer1
      1, 1, 1, 0, 1, 0,
      0, 1, 0, 0, 0, 1,
      1, 1, 1, 0, 0, 1, // layer2
      1, 0, 1, 0, 1, 0,
      1, 1, 1, 1, 0, 0,

      // Image 3: pattern1 starting at the 3rd column
      0, 0, 1, 0, 0, 0,  // layer1
      1, 1, 0, 1, 0, 0,
      0, 1, 0, 0, 1, 1,
      1, 1, 0, 0, 1, 0,  // layer2
      0, 0, 0, 1, 0, 0,
      1, 0, 1, 0, 0, 1,

      // Image 4: pattern2 starting at the 2nd column
      0, 0, 1, 0, 0, 0,  // layer1
      1, 1, 1, 1, 0, 0,
      0, 0, 1, 0, 1, 1,
      1, 1, 1, 1, 1, 0,  // layer2
      0, 1, 0, 1, 0, 0,
      1, 1, 1, 1, 0, 1,

      // Image 5: pattern1 starting at the 4th column
      0, 0, 1, 1, 0, 0,  // layer1
      1, 1, 1, 0, 1, 0,
      0, 0, 1, 0, 0, 1,
      1, 1, 1, 0, 0, 1,  // layer2
      0, 1, 0, 0, 1, 0,
      1, 1, 1, 1, 0, 0,

      // Image 6: pattern2 starting at the 1st column
      0, 1, 0, 0, 0, 0,  // layer1
      1, 1, 1, 1, 0, 0,
      0, 1, 0, 0, 1, 1,
      1, 1, 1, 1, 1, 0,  // layer2
      1, 0, 1, 1, 0, 0,
      1, 1, 1, 1, 0, 1,

      // Image 7: garbage
      0, 1, 0, 0, 0, 0,  // layer1
      1, 0, 0, 1, 0, 0,
      0, 1, 0, 0, 1, 1,
      1, 1, 1, 1, 1, 0,  // layer2
      1, 1, 0, 1, 0, 0,
      1, 1, 1, 1, 0, 1,

      // Image 8: garbage
      0, 0, 1, 0, 0, 0,  // layer1
      1, 1, 1, 1, 0, 0,
      1, 1, 1, 0, 1, 1,
      1, 1, 1, 1, 1, 1,  // layer2
      0, 0, 1, 0, 0, 0,
      1, 0, 1, 0, 0, 1,
  });
  DeviceMatrix training_y(
      DeviceMatrix(8, 2, (float[]) {
          1, 1,
          1, 1,
          1, 0,
          0, 1,
          1, 0,
          0, 1,
          0, 0,
          0, 0,
      }).T());

  std::shared_ptr<LayerStack> stack = std::make_shared<LayerStack>();
  stack->AddLayer(std::make_shared<ConvolutionalLayer>(
        2, 3, 3,
        0, 2, 1));
  // Output is supposed to be a bunch of row-vectors of length
  // 4.
  stack->AddLayer(std::make_shared<SigmoidLayer>());
  stack->AddLayer(std::make_shared<ReshapeLayer>(1, 4, 2));
  stack->AddLayer(std::make_shared<FullyConnectedLayer>(8, 2));
  stack->AddLayer(std::make_shared<SigmoidLayer>());
  Model model(
      stack,
      std::make_shared<ErrorLayer>());

  model.Train(training_x, training_y, 100, 5);


  //model.Evaluate(test_x, test_y);

}
