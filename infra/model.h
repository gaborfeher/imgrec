#ifndef _INFRA_MODEL_H_
#define _INFRA_MODEL_H_

#include <memory>

#include "cnn/layer.h"

class DataSet;
class Matrix;
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

  // Prevent copy and assignment.
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

 private:
  bool logging_;
  std::shared_ptr<LayerStack> model_;
  std::shared_ptr<ErrorLayer> error_;

  void RunPhase(const DataSet& data_set, Layer::Phase phase);
};


#endif  // _INFRA_MODEL_H_
