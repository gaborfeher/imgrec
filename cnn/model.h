#ifndef _CNN_MODEL_H_
#define _CNN_MODEL_H_

#include <memory>
#include <vector>

class DeviceMatrix;
class ErrorLayer;
class Layer;
class LayerStack;

class Model {
 public:
  Model(std::shared_ptr<Layer> model, std::shared_ptr<ErrorLayer> error);

  void Train(
      const DeviceMatrix& training_x,
      const DeviceMatrix& training_y,
      int iterations,
      float rate,
      std::vector<float>* error_hist);
  void Evaluate(
      const DeviceMatrix& test_x,
      const DeviceMatrix& test_y,
      float* error);

 private:
  std::shared_ptr<Layer> model_;
  std::shared_ptr<ErrorLayer> error_;
  std::shared_ptr<LayerStack> combined_;
};


#endif  // _CNN_MODEL_H_
