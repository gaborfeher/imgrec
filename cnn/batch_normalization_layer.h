#ifndef _BATCH_NORMALIZATION_LAYER_H_
#define _BATCH_NORMALIZATION_LAYER_H_

#include "cnn/layer.h"

class DeviceMatrix;

// https://arxiv.org/pdf/1502.03167.pdf
class BatchNormalizationLayer : public Layer {
 public:
  BatchNormalizationLayer();
  virtual void Forward(const DeviceMatrix& input);
  virtual void Backward(const DeviceMatrix& output_gradient);
  virtual void ApplyGradient(float learn_rate);

  virtual bool BeginPhase(Phase phase, int phase_sub_id);
  virtual void EndPhase(Phase phase, int phase_sub_id);
 private:
  int num_layers_per_sample_;  // 0 means each sample is a row
  DeviceMatrix beta_;
  DeviceMatrix beta_gradient_;
  DeviceMatrix gamma_;
  DeviceMatrix gamma_gradient_;

  // Intermediate values shared between Forward and Backward.
  DeviceMatrix mean_;
  DeviceMatrix shifted_;
  DeviceMatrix variance_;
  DeviceMatrix variance_e_;
  DeviceMatrix sqrt_variance_e_;
  DeviceMatrix normalized_;

  int num_samples_;

  Phase phase_;
  DeviceMatrix global_mean_;
  DeviceMatrix global_variance_;

};

#endif
