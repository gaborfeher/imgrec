#ifndef _INFRA_LOGGER_H_
#define _INFRA_LOGGER_H_

#include <fstream>
#include <chrono>
#include <memory>
#include <string>

#include "cnn/layer.h"
#include "cnn/layer_stack.h"

class Clock {
 public:
  Clock();
  void Start();
  void Stop();
  void Resume();
  float ElapsedSeconds() const;

 private:
  bool running_;
  std::chrono::steady_clock::time_point last_start_;
  std::chrono::steady_clock::duration elapsed_;
};

class Logger {
 public:
  Logger(int log_level);
  Logger(int log_level, const std::string& log_dir);

  void LogTrainingStart(int num_params);
  void LogTrainingEnd();

  void LogPhaseStart(Layer::Phase phase, int sub_id);
  void LogPhaseEnd(Layer::Phase phase, int sub_id);

  void LogMinibatchStart();
  void LogMinibatchEnd(
      int epoch, int batch,
      float error,
      float accuracy);

  void LogEpochAverage(
      int epoch,
      float error,
      float accuracy);
  void LogEpochTrainEval(
      float error,
      float accuracy);
  void LogEpochValidationEval(
      float error,
      float accuracy);
  void FinishEpochLine();

  void LogEvaluation(float error, float accuracy);

  void SaveModel(int epoch, std::shared_ptr<LayerStack> model);

 private:
  int log_level_;
  std::string log_dir_;
  std::shared_ptr<std::ofstream> summary_log_;
  std::shared_ptr<std::ofstream> detail_log_;

  void PrintBigPass(
    const std::string& color_code,
    float error, float accuracy);

  Clock training_clock_;
  Clock minibatch_clock_;
};



#endif  // _INFRA_LOGGER_H_
