#include "cnn/convolutional_layer.h"

ConvolutionalLayer::ConvolutionalLayer(
    int num_filters, int filter_width, int filter_height,
    int padding, int layers_per_image, int stride) :
        padding_(padding),
        layers_per_image_(layers_per_image),
        stride_(stride),
        filters_(filter_width, filter_height, num_filters),
        filters_gradients_(filter_width, filter_height, num_filters)
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
  input_gradients_ = output_gradients
      .AddPadding(padding_)
      .Convolution(filters_.Rot180(), layers_per_image_, stride_);  // (22) TODO is f encoded in delta for me?
  filters_gradients_ = input_
      .Rot180()
      .Convolution(output_gradients, layers_per_image_, 1);  // (14)
}

void ConvolutionalLayer::ApplyGradient(float learn_rate) {
  filters_ = filters_.Add(filters_gradients_.Multiply(-learn_rate));
}
