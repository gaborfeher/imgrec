#include "cnn/layer_test_base.h"

#include <limits>

#include "gtest/gtest.h"

#include "cnn/error_layer.h"
#include "cnn/layer_stack.h"
#include "linalg/matrix.h"
#include "linalg/matrix_test_util.h"

Matrix ComputeNumericGradient(
    const Matrix& x0,
    std::function< float (const Matrix&) > runner
) {

  Matrix result(x0.rows(), x0.cols(), x0.depth());

  float delta = 0.001f;  // I am not super-happy that this is a carefully-tuned value to make all the test pass.
  for (int k = 0; k < x0.depth(); k++) {
    for (int i = 0; i < x0.rows(); i++) {
      for (int j = 0; j < x0.cols(); j++) {
        Matrix x1(x0.DeepCopy());
        x1.SetValue(i, j, k, x0.GetValue(i, j, k) + delta);
        float error1 = runner(x1);
        x1.SetValue(i, j, k, x0.GetValue(i, j, k) - delta);
        float error2 = runner(x1);

        result.SetValue(i, j, k, (error1 - error2) / (2 * delta) );
      }
    }
  }

  return result;
}

void ParameterGradientCheck(
  std::shared_ptr<LayerStack> stack,
  const Matrix& input,
  const Matrix& param,
  std::function< void (const Matrix&) > set_param,
  std::function< Matrix() > get_param_grad,
  float absolute_diff,
  float percentage_diff) {

  set_param(param);
  stack->Forward(input);
  stack->Backward(Matrix());
  Matrix analytic_grad = get_param_grad();

  Matrix numeric_grad = ComputeNumericGradient(
      param,
      [set_param, &stack, input] (const Matrix& x) -> float {
        set_param(x);
        stack->Forward(input);
        return stack->GetLayer<ErrorLayer>(-1)->GetError();
      });

  // analytic_grad.Print(); numeric_grad.Print();
  ExpectMatrixEquals(
      analytic_grad,
      numeric_grad,
      absolute_diff,
      percentage_diff);
}

void InputGradientCheck(
  std::shared_ptr<LayerStack> stack,
  const Matrix& input,
  float absolute_diff,
  float percentage_diff) {

  stack->Forward(input);
  stack->Backward(Matrix());
  Matrix analytic_grad = stack->input_gradient();

  Matrix numeric_grad = ComputeNumericGradient(
      input,
      [&stack, input] (const Matrix& x) -> float {
        stack->Forward(x);
        return stack->GetLayer<ErrorLayer>(-1)->GetError();
      });

  // analytic_grad.Print(); numeric_grad.Print();
  ExpectMatrixEquals(
      analytic_grad,
      numeric_grad,
      absolute_diff,
      percentage_diff);
}
