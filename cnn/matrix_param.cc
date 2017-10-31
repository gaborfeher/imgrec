#include "cnn/matrix_param.h"

#include "linalg/matrix.h"

MatrixParam::MatrixParam(int rows, int cols, int depth) :
    value(rows, cols, depth),
    gradient(rows, cols, depth) {}

void MatrixParam::ApplyGradient(float learn_rate, float lambda) {
  value = value
      .Add(gradient.Multiply(-learn_rate));  // SGD
  if (lambda > 0.0f) {
    value = value.Multiply(1.0 - lambda);  // regularize
  }
}

int MatrixParam::NumParameters() const {
  return value.size();
}
