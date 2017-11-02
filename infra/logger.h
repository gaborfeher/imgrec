#ifndef _INFRA_LOGGER_H_
#define _INFRA_LOGGER_H_

#include <chrono>

#include "cnn/layer.h"

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

 private:
  int log_level_;

  Clock training_clock_;
  Clock minibatch_clock_;
};



#endif  // _INFRA_LOGGER_H_