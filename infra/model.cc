#include "infra/model.h"

#include <iostream>
#include <chrono>

#include "cnn/error_layer.h"
#include "cnn/layer_stack.h"
#include "cnn/matrix_param.h"
#include "infra/data_set.h"
#include "linalg/matrix.h"
#include "util/random.h"

void Initialize(std::shared_ptr<LayerStack> model, int random_seed) {
  Random random(random_seed);
  model->Initialize(&random);
}

Model::Model(std::shared_ptr<LayerStack> model, int random_seed) :
    log_level_(0),
    model_(model),
    error_(model->GetLayer<ErrorLayer>(-1)) {
  Initialize(model, random_seed);
}

Model::Model(std::shared_ptr<LayerStack> model, int random_seed, int log_level) :
    log_level_(log_level),
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
    const GradientInfo& gradient_info) {
  Train(data_set, epochs, gradient_info, NULL);
}

void Model::Train(
    const DataSet& data_set,
    int epochs,
    const GradientInfo& gradient_info,
    const DataSet* validation_set) {
  using std::chrono::system_clock;
  using std::chrono::duration_cast;
  using std::chrono::milliseconds;

  if (log_level_ >= 1) {
    std::cout << "Training model with " << model_->NumParameters() << " parameters" << std::endl;
  }
  system_clock::time_point training_start = system_clock::now();
  RunPhase(data_set, Layer::PRE_TRAIN_PHASE);
  model_->BeginPhase(Layer::TRAIN_PHASE, 0);
  GradientInfo grad_inf_copy = gradient_info;
  grad_inf_copy.iteration = 0;
  for (int i = 0; i < epochs; ++i) {
    float total_error = 0.0f;
    float total_accuracy = 0.0f;
    for (int j = 0; j < data_set.NumBatches(); ++j) {
      grad_inf_copy.iteration += 1;
      system_clock::time_point minibatch_start = system_clock::now();

      ForwardPass(data_set, j);
      total_error += error_->GetError();
      total_accuracy += error_->GetAccuracy();
      Matrix dummy;
      model_->Backward(dummy);
      model_->ApplyGradient(grad_inf_copy);
      system_clock::time_point minibatch_end = system_clock::now();
      float minibatch_duration =
          duration_cast<milliseconds>(minibatch_end - minibatch_start).count()
          / 1000.0f;
      if (log_level_ >= 2) {
        std::cout << "epoch " << i << " batch " << j
            << " (time= " << minibatch_duration << "s)"
            << " error= " << error_->GetError() / data_set.MiniBatchSize()
            << " accuracy= " << 100.0 * error_->GetAccuracy() << "%"
            << std::endl;
      }
    }
    float avg_error = total_error / data_set.NumBatches() / data_set.MiniBatchSize();
    float avg_accuracy = total_accuracy / data_set.NumBatches();
    if (log_level_ >= 1) {
      std::cout << "epoch " << i
          << " error= " << avg_error
          << " accuracy= " << 100.0 * avg_accuracy << "%"
          << std::endl;
    }
    if (validation_set) {
      model_->EndPhase(Layer::TRAIN_PHASE, 0);
      RunPhase(data_set, Layer::POST_TRAIN_PHASE);
      float tmp1, tmp2;
      Evaluate(*validation_set, &tmp1, &tmp2);
      model_->BeginPhase(Layer::TRAIN_PHASE, 0);
    }
  }
  model_->EndPhase(Layer::TRAIN_PHASE, 0);
  RunPhase(data_set, Layer::POST_TRAIN_PHASE);

  system_clock::time_point training_end = system_clock::now();
  float training_duration = duration_cast<std::chrono::milliseconds>(training_end - training_start).count() / 1000.0f;
  if (log_level_ >= 1) {
    std::cout << "Training time: " << training_duration << "s" << std::endl;
  }
}

void Model::RunPhase(
    const DataSet& data_set,
    Layer::Phase phase) {
  int phase_sub_id = 0;
  while (model_->BeginPhase(phase, phase_sub_id)) {
    if (log_level_ >= 2) {
      std::cout << "Running phase " << phase << " " << phase_sub_id << std::endl;
    }
    for (int j = 0; j < data_set.NumBatches(); ++j) {
      ForwardPass(data_set, j);
    }
    model_->EndPhase(phase, phase_sub_id);
    if (log_level_ >= 2) {
      std::cout << "Done: phase " << phase << " " << phase_sub_id << std::endl;
    }
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
  if (log_level_ >= 1) {
    std::cout << "evaluation"
        << " error= " << *error
        << " accuracy= " << 100.0 * *accuracy << "%"
        << std::endl;
  }
  model_->EndPhase(Layer::INFER_PHASE, 0);
}
