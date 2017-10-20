#ifndef _INFRA_MODEL_H_
#define _INFRA_MODEL_H_

#include <memory>

#include "cnn/layer.h"

class DataSet;
class DeviceMatrix;
class ErrorLayer;
class LayerStack;
class Random;

class Model {
 public:
  // The last layer of model is assumed to be an ErrorLayer.
  Model(std::shared_ptr<LayerStack> model, int random_seed);
  Model(std::shared_ptr<LayerStack> model, int random_seed, bool logging);

  void Train(
      const DataSet& data_set,
      int epochs,
      float learn_rate,
      float regularization_lambda);
  void Evaluate(
      const DataSet& data_set,
      float* error,
      float* accuracy);

 private:
  bool logging_;
  std::shared_ptr<LayerStack> model_;
  std::shared_ptr<ErrorLayer> error_;

  void RunTrainingPhase(const DataSet& data_set, Layer::TrainingPhase phase);
};


#endif  // _INFRA_MODEL_H_
