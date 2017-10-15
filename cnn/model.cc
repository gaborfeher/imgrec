#include "linalg/device_matrix.h"

#include <iostream>

#include "cnn/data_set.h"
#include "cnn/error_layer.h"
#include "cnn/layer_stack.h"
#include "cnn/model.h"

Model::Model(std::shared_ptr<LayerStack> model) :
    logging_(false),
    model_(model),
    error_(model->GetLayer<ErrorLayer>(-1)) {
}

Model::Model(std::shared_ptr<LayerStack> model, bool logging) :
    logging_(logging),
    model_(model),
    error_(model->GetLayer<ErrorLayer>(-1)) {
}

void Model::Train(
    const DataSet& data_set,
    int epochs,
    float learn_rate,
    float regularization_lambda,
    std::vector<float>* error_hist) {
  for (int i = 0; i < epochs; ++i) {
    float total_error = 0.0f;
    float total_accuracy = 0.0f;
    for (int j = 0; j < data_set.NumBatches(); ++j) {
      error_->SetExpectedValue(data_set.GetBatchOutput(j));
      model_->Forward(data_set.GetBatchInput(j));
      total_error += error_->GetError();
      total_accuracy += error_->GetAccuracy();
      DeviceMatrix dummy;
      model_->Backward(dummy);
      model_->ApplyGradient(learn_rate);
      model_->Regularize(regularization_lambda);
      // std::cout << "epoch " << i << " batch " << j << " error= " << error_->GetError() << std::endl;
    }
    float avg_error = total_error / data_set.NumBatches();
    float avg_accuracy = total_accuracy / data_set.NumBatches();
    error_hist->push_back(avg_error);
    if (logging_) {
      std::cout << "epoch " << i
          << " error= " << avg_error
          << " accuracy= " << 100.0 * avg_accuracy << "%"
          << std::endl;
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
