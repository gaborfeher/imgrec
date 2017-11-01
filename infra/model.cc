#include "infra/model.h"

#include <iomanip>
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

void PrintBigPass(
    const std::string& color_code,
    float error, float accuracy) {
  std::cout
      << " [e="
      << std::fixed << std::setw(6) << std::setprecision(4)
      << error
      << "] "
      << "\033[1;" << color_code << "m"
      << std::fixed << std::setw(6) << std::setprecision(2)
      << 100.0 * accuracy << "%"
      << "\033[0m"
      << std::flush;
}

void PrintSmallPass(
    int epoch, int batch,
    float duration,
    float error,
    float accuracy) {
  std::string color_code = "36";
  std::cout << std::fixed;
  std::cout
      << "epoch "
      << std::setw(3) << epoch
      << " batch "
      << std::setw(3) << batch
      << " (time= "
      << std::setw(6) << std::setprecision(4) << duration
      << "s)"
      << " error= "
      << std::setw(6) << std::setprecision(4) << error
      << " accuracy= "
      << "\033[1;" << color_code << "m"
      << std::setw(6) << std::setprecision(2) << 100.0 * accuracy
      << "%" << "\033[0m"
      << std::endl;
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
    std::cout                       
        << "              \033[1;34m TRAIN AVERAGE \033[0m "
        << "      \033[1;31m TRAIN EVAL\033[0m "
        << "  \033[1;32m VALIDATION EVAL \033[0m "
        << std::endl;
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
      if (log_level_ >= 2) {
        float minibatch_duration =
            duration_cast<milliseconds>(minibatch_end - minibatch_start).count()
            / 1000.0f;
        PrintSmallPass(
            i, j,
            minibatch_duration,
            error_->GetError() / data_set.MiniBatchSize(),
            error_->GetAccuracy());
      }
    }
    if (log_level_ >= 1) {
      float avg_error = total_error / data_set.NumBatches() / data_set.MiniBatchSize();
      float avg_accuracy = total_accuracy / data_set.NumBatches();
      std::cout << "EPOCH " << std::setw(3) << i;
      PrintBigPass("34", avg_error, avg_accuracy);
    }
    if (validation_set) {
      model_->EndPhase(Layer::TRAIN_PHASE, 0);
      RunPhase(data_set, Layer::POST_TRAIN_PHASE);
      float err, acc;
      Evaluate0(data_set, &err, &acc);
      PrintBigPass("31", err, acc);
      Evaluate0(*validation_set, &err, &acc);
      PrintBigPass("32", err, acc);
      model_->BeginPhase(Layer::TRAIN_PHASE, 0);
      std::cout << std::endl;
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
       std::cout << "Done phase " << phase << " " << phase_sub_id << std::endl;
    }
    phase_sub_id++;
  }
}

void Model::Evaluate0(
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
}

void Model::Evaluate(
    const DataSet& data_set,
    float* error,
    float* accuracy) {
  Evaluate0(data_set, error, accuracy);
  if (log_level_ >= 1) {
    std::cout << "EVALUATION ";
    PrintBigPass("32", *error, *accuracy);
    std::cout << std::endl;
  }
  model_->EndPhase(Layer::INFER_PHASE, 0);
}

