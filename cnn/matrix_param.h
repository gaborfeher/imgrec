#ifndef _CNN_MATRIX_PARAM_H_
#define _CNN_MATRIX_PARAM_H_

#include "linalg/matrix.h"

class MatrixParam {
 public:
  Matrix value;
  Matrix gradient;

  MatrixParam() {}
  MatrixParam(int rows, int cols, int depth);

  void ApplyGradient(float learn_rate, float lambda);
  int NumParameters() const;
};

#endif
