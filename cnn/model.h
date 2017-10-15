#ifndef _CNN_MODEL_H_
#define _CNN_MODEL_H_

#include <memory>
#include <vector>

class DataSet;
class DeviceMatrix;
class ErrorLayer;
class Layer;
class LayerStack;

class Model {
 public:
  // The last layer of model is assumed to be an ErrorLayer.
  Model(std::shared_ptr<LayerStack> model);
  Model(std::shared_ptr<LayerStack> model, bool logging);

  void Train(
      const DataSet& data_set,
      int epochs,
      float learn_rate,
      float regularization_lambda,
      std::vector<float>* error_hist);
  void Evaluate(
      const DeviceMatrix& test_x,
      const DeviceMatrix& test_y,
      float* error);

 private:
  bool logging_;
  std::shared_ptr<LayerStack> model_;
  std::shared_ptr<ErrorLayer> error_;
};


#endif  // _CNN_MODEL_H_
