#include <iostream>

#include "cnn/convolutional_layer.h"

ConvolutionalLayer::ConvolutionalLayer(
    int num_filters, int filter_rows, int filter_cols,
    int padding, int layers_per_image, int stride) :
        padding_(padding),
        layers_per_image_(layers_per_image),
        stride_(stride),
        filters_(filter_rows, filter_cols, num_filters * layers_per_image),
        filters_gradients_(filter_rows, filter_cols, num_filters * layers_per_image)
{}

void ConvolutionalLayer::Forward(const DeviceMatrix& input) {
  input_ = input;
  output_ = input
      .AddPadding(padding_, padding_)
      .Convolution(filters_, layers_per_image_, stride_);
}

void ConvolutionalLayer::Backward(const DeviceMatrix& output_gradients) {
  // Below is some guesswork based on:
  // (I have no idea if doing convolution in batches breaks that
  // or not.)
  // http://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/

  // (22)
  input_gradients_ = output_gradients
      .AddPadding(filters_.rows() - 1, filters_.cols() - 1)
      .Convolution(
          filters_.Rot180(),
          1,  // (Assuming batch of 1 image.) Each filter layer is considered a separate filter, which will create an input_gradients_ layer.
          stride_);

  int num_filters = filters_.depth() / layers_per_image_;
  // (14)
  filters_gradients_ = output_gradients
      .AddPadding(filters_.rows() - 1, filters_.cols() - 1)
      .Convolution(
          input_,
          1,  // (Assuming batch of 1 image.) Each filter layer is conisdered a separate filter, which will create a filter_gradients_ layer.
          1
      ).Rot180();
}

void ConvolutionalLayer::ApplyGradient(float learn_rate) {
  filters_ = filters_.Add(filters_gradients_.Multiply(-learn_rate));
}
