#ifndef _INFRA_TRAINER_H_
#define _INFRA_TRAINER_H_

#include <memory>

#include "cnn/layer.h"

class DataSet;
class Matrix;
class ErrorLayer;
class LayerStack;
class Logger;
class Random;

class Trainer {
 public:
  // The last layer of model is assumed to be an ErrorLayer.
  Trainer(std::shared_ptr<LayerStack> model);
  Trainer(
      std::shared_ptr<LayerStack> model,
      std::shared_ptr<Logger> logger);

  void Train(
      const DataSet& data_set,
      int epochs,
      const GradientInfo& gradient_info,
      const DataSet* validation_set);
  void Train(
      const DataSet& data_set,
      int epochs,
      const GradientInfo& gradient_info);
  void Evaluate(
      const DataSet& data_set,
      float* error,
      float* accuracy);

  // Prevent copy and assignment.
  Trainer(const Trainer&) = delete;
  Trainer& operator=(const Trainer&) = delete;

 private:
  std::shared_ptr<LayerStack> model_;
  std::shared_ptr<ErrorLayer> error_;
  std::shared_ptr<Logger> logger_;

  void RunPhase(const DataSet& data_set, Layer::Phase phase);
  void ForwardPass(const DataSet& data_set, int batch_id);
  void Evaluate0(
      const DataSet& data_set,
      float* error,
      float* accuracy);
};


#endif  // _INFRA_TRAINER_H_
