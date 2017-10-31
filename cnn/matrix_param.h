#ifndef _CNN_MATRIX_PARAM_H_
#define _CNN_MATRIX_PARAM_H_

#include "linalg/matrix.h"

struct GradientInfo {
  GradientInfo(int iteration, float learn_rate, float lambda) :
      iteration(iteration),
      learn_rate(learn_rate),
      lambda(lambda) {}

  int iteration;
  float learn_rate;
  float lambda;
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

 private:
  int mode_;
};

#endif
