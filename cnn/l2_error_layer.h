#ifndef _CNN_L2_ERROR_LAYER_H_
#define _CNN_L2_ERROR_LAYER_H_

#include "cnn/error_layer.h"
#include "linalg/matrix.h"

// Note: this implements L2^2, because its gradient is "nicely"
// going to zero near its zero value, unlike L2, which just
// jumps to zero. (The main motivation for this change was
// to make overfitting in BatchNormalizationLayerTest.TrainTest_*
// work.)
class L2ErrorLayer : public ErrorLayer {
 public:
  L2ErrorLayer();
  virtual void SetExpectedValue(const Matrix& expected_value);
  virtual void Forward(const Matrix& input);
  virtual void Backward(const Matrix& output_gradient);
  virtual float GetAccuracy() const;

 private:
  Matrix expected_value_;
};

#endif  // _CNN_L2_ERROR_LAYER_H_
