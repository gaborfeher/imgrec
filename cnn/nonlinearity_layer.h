#ifndef _CNN_NONLINEARITY_LAYER_H_
#define _CNN_NONLINEARITY_LAYER_H_

#include <utility>

#include "cnn/layer.h"
#include "linalg/matrix.h"

namespace cereal {
class PortableBinaryOutputArchive;
class PortableBinaryInputArchive;
class access;
}

namespace activation_functions {

typedef std::pair<::matrix_mappers::Map1Func, ::matrix_mappers::Map1Func> ActivationFunc;

ActivationFunc Sigmoid();
ActivationFunc ReLU();
ActivationFunc LReLU();

}  // namespace activation_functions

class NonlinearityLayer : public Layer {
 public:
  explicit NonlinearityLayer(
      ::activation_functions::ActivationFunc activation);
  virtual std::string Name() const;
  virtual void Forward(const Matrix& input);
  virtual void Backward(const Matrix& output_gradient);

  // serialization/deserialization
  void save(cereal::PortableBinaryOutputArchive& ar) const;
  void load(cereal::PortableBinaryInputArchive& ar);

 private:
  NonlinearityLayer() {}  // for cereal
  friend class cereal::access;

  ::matrix_mappers::Map1Func activation_function_;
  ::matrix_mappers::Map1Func activation_function_gradient_;
};


#endif  // _CNN_NONLINEARITY_LAYER_H_
