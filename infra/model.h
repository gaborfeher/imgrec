#ifndef _INFRA_MODEL_H_
#define _INFRA_MODEL_H_

#include <memory>

#include "cnn/layer.h"

class DataSet;
class Matrix;
class ErrorLayer;
class LayerStack;
class Logger;
class Random;

class Model {
 public:
  // The last layer of model is assumed to be an ErrorLayer.
  Model(
      std::shared_ptr<LayerStack> model,
      int random_seed);
  Model(
      std::shared_ptr<LayerStack> model,
      int random_seed,
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
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

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


#endif  // _INFRA_MODEL_H_
