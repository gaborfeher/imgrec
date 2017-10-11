#include "cnn/convolutional_layer.h"

#include <cassert>  // TODO: release-mode assert
#include <iostream>

ConvolutionalLayer::ConvolutionalLayer(
    int num_filters, int filter_rows, int filter_cols,
    int padding, int layers_per_image, int stride) :
        padding_(padding),
        layers_per_image_(layers_per_image),
        stride_(stride),
        filters_(filter_rows, filter_cols, num_filters * layers_per_image),
        filters_gradients_(filter_rows, filter_cols, num_filters * layers_per_image)
{
  assert(stride_ == 1);  // Backprop doesn't support other values uet.
  assert(padding_ == 0);  // Backprop doesn't support other values yet.
}

void ConvolutionalLayer::Forward(const DeviceMatrix& input) {
  input_ = input;
  output_ = input
      .AddPadding(padding_, padding_)
      .Convolution(filters_, layers_per_image_, stride_);
}

void ConvolutionalLayer::Backward(const DeviceMatrix& output_gradients) {
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

  int num_filters = filters_.depth() / layers_per_image_;
  int num_input_images = input_.depth() / layers_per_image_;

  // (22)
  input_gradients_ = output_gradients
      .AddPadding(filters_.rows() - 1, filters_.cols() - 1)
      .Convolution(
          filters_
              .ReorderLayers(1, layers_per_image_)
              .Rot180(),
          num_filters,
          1);
  // Layer-wise this means:
  //  output_gradients_ is
  //     img1-filter1, img1-filter2
  //     img2-filter1, img2-filter2
  //  filters_ after reordering is
  //     imglayer1-filter1, imglayer1-filter2
  //     imglayer2-filter1, imglayer2-filter2
  //     imglayer3-filter1, imglayer3-filter2
  //  input_gradients_ is
  //     img1-layer1, img1-layer2, img1-layer3
  //     img2-layer1, img2-layer2, img2-layer3

  // (14)
  filters_gradients_ = output_gradients
      .ReorderLayers(1, num_filters)
      .AddPadding(filters_.rows() - 1, filters_.cols() - 1)
      .Convolution(
          input_.ReorderLayers(1, layers_per_image_),
          num_input_images,
          1
      ).Rot180();
  // Layer-wise this means:
  //  output_gradients after reordering is:
  //     img1-filter1, img2-filter1,
  //     img1-filter2, img2-filter2
  //  input_ after reordering is:
  //     img1-layer1, img2-layer1
  //     img1-layer2, img2-layer2
  //     img1-layer3, img2-layer3
  //  filters_gradients_ is:
  //     filter1-layer1, filter1-layer2, filter1-layer3
  //     filter2-layer1, filter2-layer2, filter2-layer3
}

void ConvolutionalLayer::ApplyGradient(float learn_rate) {
  filters_ = filters_.Add(filters_gradients_.Multiply(-learn_rate));
}
