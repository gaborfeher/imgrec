#include "infra/trainer.h"

#include "cnn/error_layer.h"
#include "cnn/layer_stack.h"
#include "cnn/matrix_param.h"
#include "infra/data_set.h"
#include "infra/logger.h"
#include "linalg/matrix.h"
#include "util/random.h"

Trainer::Trainer(
    std::shared_ptr<LayerStack> model,
    std::shared_ptr<Random> random) :
        model_(model),
        error_(model->GetLayer<ErrorLayer>(-1)),
        logger_(std::make_shared<Logger>(0)) {
  model->Initialize(random.get());
}

Trainer::Trainer(
    std::shared_ptr<LayerStack> model,
    std::shared_ptr<Random> random,
    std::shared_ptr<Logger> logger) :
    model_(model),
    error_(model->GetLayer<ErrorLayer>(-1)),
    logger_(logger) {
  model->Initialize(random.get());
}

void Trainer::ForwardPass(const DataSet& data_set, int batch_id) {
  error_->SetExpectedValue(data_set.GetBatchOutput(batch_id));
  model_->Forward(data_set.GetBatchInput(batch_id));
}

void Trainer::Train(
    const DataSet& data_set,
    int epochs,
    const GradientInfo& gradient_info) {
  Train(data_set, epochs, gradient_info, NULL);
}

void Trainer::Train(
    const DataSet& data_set,
    int epochs,
    const GradientInfo& gradient_info,
    const DataSet* validation_set) {
  logger_->LogTrainingStart(model_->NumParameters());
  RunPhase(data_set, Layer::PRE_TRAIN_PHASE);
  model_->BeginPhase(Layer::TRAIN_PHASE, 0);
  GradientInfo grad_inf_copy = gradient_info;
  grad_inf_copy.iteration = 0;
  for (int i = 0; i < epochs; ++i) {
    float total_error = 0.0f;
    float total_accuracy = 0.0f;
    for (int j = 0; j < data_set.NumBatches(); ++j) {
      grad_inf_copy.iteration += 1;
      logger_->LogMinibatchStart();
      ForwardPass(data_set, j);
      total_error += error_->GetError();
      total_accuracy += error_->GetAccuracy();
      Matrix dummy;
      model_->Backward(dummy);
      model_->ApplyGradient(grad_inf_copy);
      logger_->LogMinibatchEnd(
          i, j,
          error_->GetError() / data_set.MiniBatchSize(),
          error_->GetAccuracy());
    }
    logger_->LogEpochAverage(
        i,
        total_error / data_set.NumBatches() / data_set.MiniBatchSize(),
        total_accuracy / data_set.NumBatches());
    if (validation_set) {
      // Compute additional stats.
      model_->EndPhase(Layer::TRAIN_PHASE, 0);
      RunPhase(data_set, Layer::POST_TRAIN_PHASE);
      float err, acc;
      Evaluate0(data_set, &err, &acc);
      logger_->LogEpochTrainEval(err, acc);
      Evaluate0(*validation_set, &err, &acc);
      logger_->LogEpochValidationEval(err, acc);
      model_->BeginPhase(Layer::TRAIN_PHASE, 0);
    } else {
      logger_->FinishEpochLine();
    }
  }
  model_->EndPhase(Layer::TRAIN_PHASE, 0);
  if (validation_set == NULL) {
    // If not NULL, then POST_TRAIN_PHASE was already run above.
    RunPhase(data_set, Layer::POST_TRAIN_PHASE);
  }
  logger_->LogTrainingEnd();
}

void Trainer::RunPhase(
    const DataSet& data_set,
    Layer::Phase phase) {
  int phase_sub_id = 0;
  while (model_->BeginPhase(phase, phase_sub_id)) {
    logger_->LogPhaseStart(phase, phase_sub_id);
    for (int j = 0; j < data_set.NumBatches(); ++j) {
      ForwardPass(data_set, j);
    }
    model_->EndPhase(phase, phase_sub_id);
    logger_->LogPhaseEnd(phase, phase_sub_id);
    phase_sub_id++;
  }
}

void Trainer::Evaluate0(
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
  model_->EndPhase(Layer::INFER_PHASE, 0);
}

void Trainer::Evaluate(
    const DataSet& data_set,
    float* error,
    float* accuracy) {
  Evaluate0(data_set, error, accuracy);
  logger_->LogEvaluation(*error, *accuracy);
}

