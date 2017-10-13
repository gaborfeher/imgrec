#include "linalg/device_matrix.h"

#include <iostream>

#include "cnn/data_set.h"
#include "cnn/error_layer.h"
#include "cnn/layer_stack.h"
#include "cnn/model.h"

Model::Model(std::shared_ptr<LayerStack> model) :
    model_(model),
    error_(model->GetLayer<ErrorLayer>(-1)) {
}

void Model::Train(
    const DataSet& data_set,
    int epochs,
    float rate,
    std::vector<float>* error_hist) {
  for (int i = 0; i < epochs; ++i) {
    for (int j = 0; j < data_set.NumBatches(); ++j) {
      error_->SetExpectedValue(data_set.GetBatchOutput(j));
      model_->Forward(data_set.GetBatchInput(j));
      error_hist->push_back(error_->GetError());
      DeviceMatrix dummy;
      model_->Backward(dummy);
      model_->ApplyGradient(rate);
    }
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
