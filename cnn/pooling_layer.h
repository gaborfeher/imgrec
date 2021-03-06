#ifndef _CNN_POOLING_LAYER_H_
#define _CNN_POOLING_LAYER_H_

#include "cnn/layer.h"

namespace cereal {
class PortableBinaryOutputArchive;
class PortableBinaryInputArchive;
class access;
}

// Splits the input into pool_rows x pool_cols subregions
// and takes the max from each of them.
class PoolingLayer : public Layer {
 public:
  PoolingLayer(int pool_rows, int pool_cols);

  virtual std::string Name() const;
  virtual void Forward(const Matrix& input);
  // TODO: if the input has multiple values tied at max,
  // then backprop makes an arbitrary choice and only sends
  // gradient back at the first one. (TODO: fix or confirm if
  // OK.)
  virtual void Backward(const Matrix& output_gradient);

  // serialization/deserialization
  void save(cereal::PortableBinaryOutputArchive& ar) const;
  void load(cereal::PortableBinaryInputArchive& ar);

 private:
  PoolingLayer() {}  // for cereal
  friend class cereal::access;

  int pool_rows_;
  int pool_cols_;
  Matrix switch_;
};

#endif  // _CNN_POOLING_LAYER_H_
