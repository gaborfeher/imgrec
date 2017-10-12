#include "linalg/device_matrix.h"

#include <iostream>

#include "cnn/error_layer.h"
#include "cnn/layer_stack.h"
#include "cnn/model.h"

Model::Model(std::shared_ptr<LayerStack> model) :
    model_(model),
    error_(model->GetLayer<ErrorLayer>(-1)) {
}

void Model::Train(
    const DeviceMatrix& training_x,
    const DeviceMatrix& training_y,
    int iterations,
    float rate,
    std::vector<float>* error_hist) {
  for (int i = 0; i < iterations; ++i) {
    error_->SetExpectedValue(training_y);
    model_->Forward(training_x);
    error_hist->push_back(error_->GetError());
    DeviceMatrix dummy;
    model_->Backward(dummy);
    model_->ApplyGradient(rate);
  }
}

void Model::Evaluate(
    const DeviceMatrix& test_x,
    const DeviceMatrix& test_y,
    float* error) {
  error_->SetExpectedValue(test_y);
  model_->Forward(test_x);
  *error = error_->GetError();
}
