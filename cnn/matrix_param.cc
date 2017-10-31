#include "cnn/matrix_param.h"

#include <cassert>
#include <iostream>

#include "linalg/matrix.h"

MatrixParam::MatrixParam(int rows, int cols, int depth) :
    value(rows, cols, depth),
    gradient(rows, cols, depth),
    mode_(1) {
  if (mode_ == 1) {
    m = Matrix(rows, cols, depth);
    v = Matrix(rows, cols, depth);
    m.Fill(0);
    v.Fill(0);
  }
}

void MatrixParam::ApplyGradient(const GradientInfo& info) {
  Matrix dx = gradient;
  if (info.lambda > 0.0f) {
    dx.Add(value.Multiply(info.lambda / info.learn_rate));
  }

  if (mode_ == 0) {
    // SGD (Stochastic Gradient Descent)
    value = value.Add(dx.Multiply(-info.learn_rate));
  } else if (mode_ == 1) {
    // ADAM (See: http://cs231n.github.io/neural-networks-3/)
    assert(info.iteration >= 1);
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8;

    m = m.Multiply(beta1).Add(dx.Multiply(1.0f - beta1));
    Matrix mt = m.Divide(
        1.0f - std::pow(beta1, info.iteration));
    v = v.Multiply(beta2).Add(
        dx
            .Map1(::matrix_mappers::Square())
            .Multiply(1.0f - beta2));
    Matrix vt = v.Divide(
        1.0f - std::pow(beta2, info.iteration));
    value = value.Add(mt
        .Multiply(-info.learn_rate)
        .ElementwiseDivide(
            vt
                .Map1(::matrix_mappers::Sqrt())
                .AddConst(eps)));
  } else {
    assert(false);
  }
}

int MatrixParam::NumParameters() const {
  return value.size();
}
