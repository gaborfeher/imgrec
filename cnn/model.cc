#include "linalg/device_matrix.h"

#include <iostream>

#include "cnn/error_layer.h"
#include "cnn/layer_stack.h"
#include "cnn/model.h"

Model::Model(std::shared_ptr<Layer> model, std::shared_ptr<ErrorLayer> error) :
    model_(model),
    error_(error) {
  combined_ = std::make_shared<LayerStack>();
  combined_->AddLayer(model_);
  combined_->AddLayer(error_);
}

void Model::Train(const DeviceMatrix& training_x, const DeviceMatrix& training_y, int iterations, float rate) {
  for (int i = 0; i < iterations; ++i) {
    error_->SetExpectedValue(training_y);
    combined_->Forward(training_x);
    error_->output().AssertDimensions(1, 1, 1);
    std::cout << "Training Error= " << error_->output().GetVector()[0] << std::endl;
    DeviceMatrix dummy;
    combined_->Backward(dummy);
    combined_->ApplyGradient(rate);
  }
}

void Model::Evaluate(const DeviceMatrix& test_x, const DeviceMatrix& test_y) {
  error_->SetExpectedValue(test_y);
  combined_->Forward(test_x);
  error_->output().AssertDimensions(1, 1, 1);
  std::cout << "Test Error= " << error_->output().GetVector()[0] << std::endl;
}
