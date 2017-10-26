#include "infra/model.h"

#include <iostream>

#include "cnn/error_layer.h"
#include "cnn/layer_stack.h"
#include "infra/data_set.h"
#include "linalg/matrix.h"
#include "util/random.h"

void Initialize(std::shared_ptr<LayerStack> model, int random_seed) {
  Random random(random_seed);
  model->Initialize(&random);
}

Model::Model(std::shared_ptr<LayerStack> model, int random_seed) :
    logging_(false),
    model_(model),
    error_(model->GetLayer<ErrorLayer>(-1)) {
  Initialize(model, random_seed);
}

Model::Model(std::shared_ptr<LayerStack> model, int random_seed, bool logging) :
    logging_(logging),
    model_(model),
    error_(model->GetLayer<ErrorLayer>(-1)) {
  Initialize(model, random_seed);
}

void Model::ForwardPass(const DataSet& data_set, int batch_id) {
  error_->SetExpectedValue(data_set.GetBatchOutput(batch_id));
  model_->Forward(data_set.GetBatchInput(batch_id));
}

void Model::Train(
    const DataSet& data_set,
    int epochs,
    float learn_rate,
    float regularization_lambda) {

  RunPhase(data_set, Layer::PRE_TRAIN_PHASE);
  model_->BeginPhase(Layer::TRAIN_PHASE, 0);
  for (int i = 0; i < epochs; ++i) {
    float total_error = 0.0f;
    float total_accuracy = 0.0f;
    for (int j = 0; j < data_set.NumBatches(); ++j) {
      ForwardPass(data_set, j);
      total_error += error_->GetError();
      total_accuracy += error_->GetAccuracy();
      Matrix dummy;
      model_->Backward(dummy);
      model_->ApplyGradient(learn_rate);
      model_->Regularize(regularization_lambda);
      // std::cout << "epoch " << i << " batch " << j << " error= " << error_->GetError() << std::endl;
    }
    float avg_error = total_error / data_set.NumBatches() / data_set.MiniBatchSize();
    float avg_accuracy = total_accuracy / data_set.NumBatches();
    if (logging_) {
      std::cout << "epoch " << i
          << " error= " << avg_error
          << " accuracy= " << 100.0 * avg_accuracy << "%"
          << std::endl;
      // model_->Print();
    }
  }
  model_->EndPhase(Layer::TRAIN_PHASE, 0);
  RunPhase(data_set, Layer::POST_TRAIN_PHASE);
}

void Model::RunPhase(
    const DataSet& data_set,
    Layer::Phase phase) {
  int phase_sub_id = 0;
  while (model_->BeginPhase(phase, phase_sub_id)) {
    for (int j = 0; j < data_set.NumBatches(); ++j) {
      ForwardPass(data_set, j);
    }
    model_->EndPhase(phase, phase_sub_id);
    phase_sub_id++;
  }
}

void Model::Evaluate(
    const DataSet& data_set,
    float* error,
    float* accuracy) {
  model_->BeginPhase(Layer::INFER_PHASE, 0);
  float total_error = 0.0f;
  float total_accuracy = 0.0f;
  for (int j = 0; j < data_set.NumBatches(); ++j) {
    error_->SetExpectedValue(data_set.GetBatchOutput(j));
    model_->Forward(data_set.GetBatchInput(j));
    total_error += error_->GetError();
    total_accuracy += error_->GetAccuracy();
  }
  *error = total_error / data_set.NumBatches() / data_set.MiniBatchSize();
  *accuracy = total_accuracy / data_set.NumBatches();
  if (logging_) {
    std::cout << "evaluation"
        << " error= " << *error
        << " accuracy= " << 100.0 * *accuracy << "%"
        << std::endl;
  }
  model_->EndPhase(Layer::INFER_PHASE, 0);
}
