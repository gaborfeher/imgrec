#ifndef _CNN_LAYER_TEST_BASE_H_
#define _CNN_LAYER_TEST_BASE_H_

#include <functional>
#include <memory>

class Matrix;
class LayerStack;

// Compare numeric and analytic estimates of the gradient of
// the parameter value controlled by |set_param|/|get_param_grad|
// in |stack|. The input of forward passes is set constant at
// |input|. The tolerance for comparison is |absolute_diff| and
// |percentage_diff|.
void ParameterGradientCheck(
  std::shared_ptr<LayerStack> stack,
  const Matrix& input,
  const Matrix& param,
  std::function< void (const Matrix&) > set_param,
  std::function< Matrix() > get_param_grad,
  float absolute_diff,
  float percentage_diff);

// Compare numeric and analytic estimates of the gradient of
// the input of the |stack|.
// The tolerance for comparison is |absolute_diff|.
void InputGradientCheck(
  std::shared_ptr<LayerStack> stack,
  const Matrix& input,
  float absolute_diff,
  float percentage_diff);

#endif  // _CNN_LAYER_TEST_BASE_H_
