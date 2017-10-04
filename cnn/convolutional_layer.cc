#include <iostream>

#include "cnn/convolutional_layer.h"

ConvolutionalLayer::ConvolutionalLayer(
    int num_filters, int filter_width, int filter_height,
    int padding, int layers_per_image, int stride) :
        padding_(padding),
        layers_per_image_(layers_per_image),
        stride_(stride),
        filters_(filter_width, filter_height, num_filters * layers_per_image),
        filters_gradients_(filter_width, filter_height, num_filters * layers_per_image)
{}

void ConvolutionalLayer::Forward(const DeviceMatrix& input) {
  input_ = input;
  output_ = input
      .AddPadding(padding_)
      .Convolution(filters_, layers_per_image_, stride_);
}

void ConvolutionalLayer::Backward(const DeviceMatrix& output_gradients) {
  // Below is some guesswork based on:
  // (I have no idea if doing convolution in batches breaks that
  // or not.)
  // http://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/

  // (22) TODO is f encoded in delta for me?
  // We add up gradients coming from the different filters.
  input_gradients_ = output_gradients
      .AddPadding(padding_)
      .Convolution(
          filters_.Rot180(),
          filters_.depth(),  // Treat it like there is one big filter for each image. This should add up the computed gradients per image.
          stride_);

  int num_filters = filters_.depth() / layers_per_image_;
  // (14)
  filters_gradients_ = input_
      .Rot180()
      .Convolution(
          output_gradients.ReorderLayers(layers_per_image_, num_filters),  // all outputs for filter1; all outputs for filter2; etc.  // TODO: avoid ReorderLayers by doing it on the fly in Convolution?
          input_.depth() / num_filters,  // Treat it like input_ is one big image, and output_gradients_ has #filters filters. TODO (this was just a guess)
          1  // TODO
      );

}

void ConvolutionalLayer::ApplyGradient(float learn_rate) {
  filters_ = filters_.Add(filters_gradients_.Multiply(-learn_rate));
}
