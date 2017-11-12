#include "infra/logger.h"

#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "cereal/archives/portable_binary.hpp"
#include "cereal/types/memory.hpp"

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
  num_runs_ = 0;
}

void Clock::Stop() {
  assert(running_ == true);
  elapsed_ += steady_clock::now() - last_start_;
  running_ = false;
  num_runs_++;
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

float Clock::AverageSeconds() const {
  assert(running_ == false);
  assert(num_runs_ > 0);
  return ElapsedSeconds() / num_runs_;
}

Logger::Logger(int log_level)
    : log_level_(log_level) {}

Logger::Logger(int log_level, const std::string& log_dir)
    : log_level_(log_level),
      log_dir_(log_dir),
      summary_log_(std::make_shared<std::ofstream>(log_dir + "/summary.txt")),
      detail_log_(std::make_shared<std::ofstream>(log_dir + "/details.txt")) {
  std::cout << "Log files:" << std::endl;
  std::cout << log_dir + "/summary.txt" << std::endl;
  std::cout << log_dir + "/details.txt" << std::endl;
}

void Logger::LogTrainingStart(int num_params) {
  if (log_level_ >= 1) {
    training_clock_.Start();
    std::cout << "Training model with " << num_params << " parameters" << std::endl;
    std::cout
        << "              \033[1;34m TRAIN AVERAGE \033[0m "
        << "      \033[1;31m TRAIN EVAL\033[0m "
        << "  \033[1;32m VALIDATION EVAL \033[0m "
        << std::endl;
    if (summary_log_) {
      *summary_log_ << "Training model with " << num_params << " parameters" << std::endl;
      *summary_log_
          << "               TRAIN AVERAGE  "
          << "       TRAIN EVAL "
          << "   VALIDATION EVAL  "
          << std::endl;
    }
  }
}

void Logger::LogTrainingEnd() {
  if (log_level_ >= 1) {
    float time = training_clock_.ElapsedSeconds();
    std::cout
        << "Training time: " << time << "s" << std::endl;
    if (summary_log_) {
      *summary_log_
          << "Training time: " << time << "s" << std::endl;
    }
  }
}

void Logger::LogMinibatchStart() {
  if (log_level_ >= 1) {
    minibatch_clock_.Start();
  }
}

void Logger::LogMinibatchEnd(
    int epoch, int batch,
    float error,
    float accuracy) {
  if (log_level_ >= 2) {
    *detail_log_ << std::fixed;
    *detail_log_
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
        << std::setw(6) << std::setprecision(2) << 100.0 * accuracy
        << std::endl;
  }
  if (log_level_ >= 3) {
    for (std::map<std::string, std::map<std::string, Clock>>::value_type item : layer_clocks_) {
      std::string pad = std::string(
          std::max(0, 36 - static_cast<int>(item.first.size())),
          '.');
      *detail_log_
          << std::fixed
          << item.first << pad << ":"
          << std::setprecision(2)
          << std::setw(8)
          << item.second["FW"].AverageSeconds() * 1000 << "ms "
          << std::setw(8)
          << item.second["BW"].AverageSeconds() * 1000 << "ms"
          << std::endl;
    }
  }
}

void Logger::PrintBigPass(
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
  if (summary_log_) {
      *summary_log_
        << " [e="
        << std::fixed << std::setw(6) << std::setprecision(4)
        << error
        << "] "
        << std::fixed << std::setw(6) << std::setprecision(2)
        << 100.0 * accuracy << "%"
        << std::flush;
  }
}

void Logger::LogEpochAverage(
    int epoch,
    float error,
    float accuracy) {
  if (log_level_ >= 1) {
    std::cout << "EPOCH " << std::setw(3) << epoch;
    if (summary_log_) {
      *summary_log_ << "EPOCH " << std::setw(3) << epoch;
    }
    PrintBigPass("34", error, accuracy);
  }
}

void Logger::FinishEpochLine() {
  if (log_level_ >= 1) {
    std::cout << std::endl;
    if (summary_log_) {
      *summary_log_ << std::endl;
    }
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
    if (summary_log_) {
      *summary_log_ << std::endl;
    }
  }
}

void Logger::LogEvaluation(float error, float accuracy) {
  if (log_level_ >= 1) {
    std::cout << "EVALUATION ";
    if (summary_log_) {
      *summary_log_ << "EVALUATION ";
    }
    PrintBigPass("32", error, accuracy);
    std::cout << std::endl;
    if (summary_log_) {
      *summary_log_ << std::endl;
    }
  }
}

void Logger::LogPhaseStart(Layer::Phase phase, int sub_id) {
  if (log_level_ >= 2) {
    *detail_log_ << "Running phase " << phase << " " << sub_id << std::endl;
  }
}

void Logger::LogPhaseEnd(Layer::Phase phase, int sub_id) {
  if (log_level_ >= 2) {
    *detail_log_ << "Finished phase " << phase << " " << sub_id << std::endl;
  }
  layer_clocks_.clear();
}

void Logger::SaveModel(int epoch, std::shared_ptr<LayerStack> model) {
  if (log_level_ >= 2) {
    std::ofstream os(
        log_dir_ + "/epoch" + std::to_string(epoch) + ".model",
        std::ios::out | std::ios::binary);
    cereal::PortableBinaryOutputArchive output(os);
    output(model);
  }
}

std::string GetClockId(
    int id,
    const std::string& name) {
  std::stringstream ss;
  ss
      << "["
      << std::fixed << std::setw(2) << std::setfill('0')
      << id
      << "]"
      << name;
  return ss.str();
}

void Logger::LogLayerStart(
    int id,
    const std::string& name,
    const std::string& op_kind) {
  if (log_level_ >= 3) {
    // If needed, then this creates a Clock with default constructor. The default constructor
    // Start()s the clock. Resume()ing a Start()ed clock just restarts it.
    layer_clocks_[GetClockId(id, name)][op_kind].Resume();
  }
}

void Logger::LogLayerFinish(
    int id,
    const std::string& name,
    const std::string& op_kind) {
  if (log_level_ >= 3) {
    layer_clocks_[GetClockId(id, name)][op_kind].Stop();
  }
}
