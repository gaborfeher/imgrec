#ifndef _CNN_NONLINEARITY_LAYER_H_
#define _CNN_NONLINEARITY_LAYER_H_

#include <utility>

#include "cnn/layer.h"
#include "linalg/device_matrix.h"

namespace activation_functions {

typedef std::pair<::matrix_mappers::MapperFunc, ::matrix_mappers::MapperFunc> ActivationFunc;

  ActivationFunc Sigmoid();

}  // namespace activation_functions

class NonlinearityLayer : public Layer {
 public:
  NonlinearityLayer(::activation_functions::ActivationFunc activation);
  virtual void Forward(const DeviceMatrix& input);
  virtual void Backward(const DeviceMatrix& output_gradients);
  virtual void ApplyGradient(float learn_rate);
 private:
  ::matrix_mappers::MapperFunc activation_function_;
  ::matrix_mappers::MapperFunc activation_function_gradient_;
};


#endif  // _CNN_NONLINEARITY_LAYER_H_
