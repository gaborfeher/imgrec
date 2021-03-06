#ifndef _CNN_RESHAPE_LAYER_H_
#define _CNN_RESHAPE_LAYER_H_

#include "cnn/layer.h"

namespace cereal {
class PortableBinaryOutputArchive;
class PortableBinaryInputArchive;
class access;
}

// Turns a matrix input into a column-vector output.
class ReshapeLayer : public Layer {
 public:
  ReshapeLayer(int unit_rows, int unit_cols, int unit_depth);
  virtual std::string Name() const;
  virtual void Forward(const Matrix& input);
  virtual void Backward(const Matrix& output_gradient); 

  // serialization/deserialization
  void save(cereal::PortableBinaryOutputArchive& ar) const;
  void load(cereal::PortableBinaryInputArchive& ar);

 private:
  ReshapeLayer() {}  // for cereal
  friend class cereal::access;

  int unit_rows_;
  int unit_cols_;
  int unit_depth_;
};

#endif  // _CNN_RESHAPE_LAYER_H_
