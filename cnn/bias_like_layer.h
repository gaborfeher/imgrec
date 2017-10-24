#ifndef _CNN_BIAS_LIKE_LAYER_H_
#define _CNN_BIAS_LIKE_LAYER_H_

#include "cnn/layer.h"

// Subclasses of this layer transform output signals of neurons.
// (The transformation is learned.) The trick is that there are
// two cases:
// 1. The output of a fully-connected layer is a rows x cols matrix,
//    where rows is the number of neurons, and cols is the number of
//    samples processed in the current minibatch.
// 2. The output of a convolutional layer is a rows x cols x depth
//    3D matrix, where rows x cols is the image size and
//    depth = number of neurons (i.e. filters) *
//            number of samples (images) processed in the current
//            minibatch.
// In both cases, elements corresponding the same neuron should
// get the same treatment.
//
// (Currently the only purpose of this abstract class is common
// documentation and common member variable naming for subclasses.)
class BiasLikeLayer : public Layer {
 public:
  BiasLikeLayer(int num_neurons, bool layered) :
      layered_(layered),
      num_neurons_(num_neurons) {}
 protected:
  bool layered_;
  int num_neurons_;
};

#endif  // _CNN_BIAS_LIKE_LAYER_H_
