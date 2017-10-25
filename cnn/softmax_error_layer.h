#ifndef _CNN_SOFTMAX_ERROR_LAYER_H_
#define _CNN_SOFTMAX_ERROR_LAYER_H_

#include "cnn/error_layer.h"
#include "linalg/matrix.h"

class SoftmaxErrorLayer : public ErrorLayer {
 public:
  SoftmaxErrorLayer();
  virtual void SetExpectedValue(const Matrix& expected_value);
  virtual void Forward(const Matrix& input);
  virtual void Backward(const Matrix& output_gradient);
  virtual float GetAccuracy() const;

 private:
  Matrix expected_value_;
};

#endif  // _CNN_SOFTMAX_ERROR_LAYER_H_
