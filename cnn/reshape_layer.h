#ifndef _CNN_RESHAPE_LAYER_H_
#define _CNN_RESHAPE_LAYER_H_

#include "cnn/layer.h"

// Turns a matrix input into a column-vector output.
class ReshapeLayer : public Layer {
 public:
  ReshapeLayer(int unit_rows, int unit_cols, int unit_depth);
  virtual void Forward(const Matrix& input);
  virtual void Backward(const Matrix& output_gradient); 
  virtual void ApplyGradient(float learn_rate);

 private:
  int unit_rows_;
  int unit_cols_;
  int unit_depth_;
};

#endif  // _CNN_RESHAPE_LAYER_H_
