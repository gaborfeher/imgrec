#ifndef _CNN_POOLING_LAYER_H_
#define _CNN_POOLING_LAYER_H_

#include "cnn/layer.h"

// Splits the input into pool_rows x pool_cols subregions
// and takes the max from each of them.
class PoolingLayer : public Layer {
 public:
  PoolingLayer(int pool_rows, int pool_cols);

  virtual void Print() const;
  virtual void Forward(const Matrix& input);
  // TODO: if the input has multiple values tied at max,
  // then backprop makes an arbitrary choice and only sends
  // gradient back at the first one. (TODO: fix or confirm if
  // OK.)
  virtual void Backward(const Matrix& output_gradient);

 private:
  int pool_rows_;
  int pool_cols_;
  Matrix switch_;
};

#endif  // _CNN_POOLING_LAYER_H_
