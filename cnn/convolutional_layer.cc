#include "cnn/convolutional_layer.h"

#include <cassert>  // TODO: release-mode assert
#include <random>
#include <iostream>

#include "util/random.h"

ConvolutionalLayer::ConvolutionalLayer(
    int num_filters, int filter_rows, int filter_cols,
    int padding, int layers_per_image) :
        padding_(padding),
        layers_per_image_(layers_per_image),
        filters_(filter_rows, filter_cols, num_filters * layers_per_image) {
}

void ConvolutionalLayer::Print() const {
  std::cout << "Convolutional Layer:" << std::endl;
  filters_.value.Print();
}

void ConvolutionalLayer::Initialize(Random* random) {
  // http://cs231n.github.io/neural-networks-2/#init
  // https://stats.stackexchange.com/questions/198840/cnn-xavier-weight-initialization
  float variance = 2.0f / (filters_.value.rows() * filters_.value.cols() * layers_per_image_);
  std::normal_distribution<float> dist(0, sqrt(variance));
  filters_.value.RandomFill(random, &dist);
}

void ConvolutionalLayer::Forward(const Matrix& input) {
  input_ = input;
  output_ = Matrix::Convolution(
      layers_per_image_,
      input, true, padding_, padding_,
      filters_.value, true, 0, 0,
      0, 0);
}

void ConvolutionalLayer::Backward(const Matrix& output_gradient) {
  // This implementation is based on the following article, plus
  // additions.
  // http://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
  // Additions:
  // - We add padding before the convolution operations, so that
  //   matrix dimensions will match. (This is not explicitly
  //   mentioned in the article.)
  // - Rotations differ because our "convolution" = their convolution
  //   + a 180 rotation on the convolution matrix.
  // - Handling of multiple input images and multiple filters
  //   is supported. (The same convolutions are executed as in
  //   the article, but there is extra juggling with the layers
  //   of inputs, outputs and filters.)

  int num_filters = filters_.value.depth() / layers_per_image_;
  int num_input_images = input_.depth() / layers_per_image_;

  // (22)
  input_gradient_ = Matrix::Convolution(
      num_filters,
      output_gradient, true, filters_.value.rows() - 1, filters_.value.cols() - 1,
      filters_.value.Rot180(), false, 0, 0,
      padding_, padding_);
  // Layer-wise this means:
  //  output_gradient_ is
  //     img1-filter1, img1-filter2
  //     img2-filter1, img2-filter2
  //  filters_ after reordering is
  //     imglayer1-filter1, imglayer1-filter2
  //     imglayer2-filter1, imglayer2-filter2
  //     imglayer3-filter1, imglayer3-filter2
  //  input_gradient_ is
  //     img1-layer1, img1-layer2, img1-layer3
  //     img2-layer1, img2-layer2, img2-layer3

  // (14)
  filters_.gradient = Matrix::Convolution(
      num_input_images,
      output_gradient, false, filters_.value.rows() - 1, filters_.value.cols() - 1,
      input_, false, padding_, padding_,
      0, 0)
          .Rot180();
  // Layer-wise this means:
  //  output_gradient after reordering is:
  //     img1-filter1, img2-filter1,
  //     img1-filter2, img2-filter2
  //  input_ after reordering is:
  //     img1-layer1, img2-layer1
  //     img1-layer2, img2-layer2
  //     img1-layer3, img2-layer3
  //  filters_gradient_ is:
  //     filter1-layer1, filter1-layer2, filter1-layer3
  //     filter2-layer1, filter2-layer2, filter2-layer3
}

void ConvolutionalLayer::ApplyGradient(float learn_rate, float lambda) {
  filters_.ApplyGradient(learn_rate, lambda);
}

int ConvolutionalLayer::NumParameters() const {
  return filters_.NumParameters();
}

