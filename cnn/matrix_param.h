#ifndef _CNN_MATRIX_PARAM_H_
#define _CNN_MATRIX_PARAM_H_

#include "linalg/matrix.h"

namespace cereal {
class PortableBinaryOutputArchive;
class PortableBinaryInputArchive;
}

struct GradientInfo {
  enum Mode {
    SGD,
    ADAM
  };

  GradientInfo(float learn_rate, float lambda, Mode mode) :
      iteration(0),  // invalid value
      learn_rate(learn_rate),
      lambda(lambda),
      mode(mode) {}

  int iteration;
  float learn_rate;
  float lambda;
  Mode mode;
};

class MatrixParam {
 public:
  Matrix value;
  Matrix gradient;

  Matrix m;
  Matrix v;

  MatrixParam() {}
  MatrixParam(int rows, int cols, int depth);

  void ApplyGradient(const GradientInfo& info);
  int NumParameters() const;

  // serialization/deserialization
  void save(cereal::PortableBinaryOutputArchive& ar) const;
  void load(cereal::PortableBinaryInputArchive& ar);
};

#endif
