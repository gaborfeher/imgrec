#include "infra/logger.h"

#include <iomanip>
#include <iostream>

using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

Clock::Clock() {
  Start();
}

void Clock::Start() {
  last_start_ = steady_clock::now();
  elapsed_ = steady_clock::duration::zero();
  running_ = true;
}

void Clock::Stop() {
  steady_clock::time_point now = steady_clock::now();
  elapsed_ += last_start_ - now;
  running_ = false;
}

void Clock::Resume() {
  last_start_ = steady_clock::now();
  running_ = true;
}

float Clock::ElapsedSeconds() const {
  steady_clock::duration elapsed = elapsed_;
  if (running_) {
    elapsed += steady_clock::now() - last_start_;
  }
  return duration_cast<milliseconds>(elapsed).count() / 1000.0f;
}

Logger::Logger(int log_level) : log_level_(log_level) {}

void Logger::LogTrainingStart(int num_params) {
  training_clock_.Start();
  if (log_level_ >= 1) {
    std::cout << "Training model with " << num_params << " parameters" << std::endl;
    std::cout
        << "              \033[1;34m TRAIN AVERAGE \033[0m "
        << "      \033[1;31m TRAIN EVAL\033[0m "
        << "  \033[1;32m VALIDATION EVAL \033[0m "
        << std::endl;
  }
}

void Logger::LogTrainingEnd() {
  if (log_level_ >= 1) {
    std::cout
        << "Training time: "
        << training_clock_.ElapsedSeconds()
        << "s" << std::endl;
  }
}

void Logger::LogMinibatchStart() {
  minibatch_clock_.Start();
}

void Logger::LogMinibatchEnd(
    int epoch, int batch,
    float error,
    float accuracy) {
  if (log_level_ >= 2) {
    std::string color_code = "36";
    std::cout << std::fixed;
    std::cout
        << "epoch "
        << std::setw(3) << epoch
        << " batch "
        << std::setw(3) << batch
        << " (time= "
        << std::setw(6) << std::setprecision(4) << minibatch_clock_.ElapsedSeconds()
        << "s)"
        << " error= "
        << std::setw(6) << std::setprecision(4) << error
        << " accuracy= "
        << "\033[1;" << color_code << "m"
        << std::setw(6) << std::setprecision(2) << 100.0 * accuracy
        << "%" << "\033[0m"
        << std::endl;
  }
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

void Logger::LogEpochAverage(
    int epoch,
    float error,
    float accuracy) {
  if (log_level_ >= 1) {
    std::cout << "EPOCH " << std::setw(3) << epoch;
    PrintBigPass("34", error, accuracy);
  }
}

void Logger::FinishEpochLine() {
  if (log_level_ >= 1) {
    std::cout << std::endl;
  }
}

void Logger::LogEpochTrainEval(
    float error,
    float accuracy) {
  if (log_level_ >= 1) {
    PrintBigPass("31", error, accuracy);
  }
}

void Logger::LogEpochValidationEval(
    float error,
    float accuracy) {
  if (log_level_ >= 1) {
    PrintBigPass("32", error, accuracy);
    std::cout << std::endl;
  }
}

void Logger::LogEvaluation(float error, float accuracy) {
  if (log_level_ >= 1) {
    std::cout << "EVALUATION ";
    PrintBigPass("32", error, accuracy);
    std::cout << std::endl;
  }
}

void Logger::LogPhaseStart(Layer::Phase phase, int sub_id) {
  if (log_level_ >= 2) {
    std::cout << "Running phase " << phase << " " << sub_id << std::endl;
  }
}

void Logger::LogPhaseEnd(Layer::Phase phase, int sub_id) {
  if (log_level_ >= 2) {
    std::cout << "Finished phase " << phase << " " << sub_id << std::endl;
  }
}
