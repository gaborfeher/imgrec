#ifndef _BATCH_NORMALIZATION_LAYER_H_
#define _BATCH_NORMALIZATION_LAYER_H_

#include "cnn/layer.h"

class DeviceMatrix;

class BatchNormalizationLayer : public Layer {
 public:
  virtual void Forward(const DeviceMatrix& input);
  virtual void Backward(const DeviceMatrix& output_gradient);

  virtual bool BeginTrainingPhase(TrainingPhase phase);
  virtual void EndTrainingPhase(TrainingPhase phase);
 private:
  DeviceMatrix beta_;
  DeviceMatrix gamma_;

};

#endif
