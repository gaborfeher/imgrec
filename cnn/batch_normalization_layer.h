#ifndef _BATCH_NORMALIZATION_LAYER_H_
#define _BATCH_NORMALIZATION_LAYER_H_

#include "cnn/bias_like_layer.h"

#include "gtest/gtest_prod.h"

class DeviceMatrix;
class Random;

// https://arxiv.org/pdf/1502.03167.pdf
class BatchNormalizationLayer : public BiasLikeLayer {
 public:
  // See BiasLikeLayer for param docs.
  // non-layered:
  explicit BatchNormalizationLayer(int num_neurons);
  // layered:
  BatchNormalizationLayer(int layer_rows, int layer_cols, int num_neurons);
  virtual void Initialize(Random*);
  virtual void Forward(const DeviceMatrix& input);
  virtual void Backward(const DeviceMatrix& output_gradient);
  virtual void ApplyGradient(float learn_rate);

  virtual bool BeginPhase(Phase phase, int phase_sub_id);
  virtual void EndPhase(Phase phase, int phase_sub_id);
 private:
  FRIEND_TEST(BatchNormalizationLayerTest, ForwardNormalization_ColumnMode);
  FRIEND_TEST(BatchNormalizationLayerTest, ForwardBetaGamma_ColumnMode);
  FRIEND_TEST(BatchNormalizationLayerTest, Forward_LayerMode);
  FRIEND_TEST(BatchNormalizationLayerTest, GradientCheck_ColumnMode);
  FRIEND_TEST(BatchNormalizationLayerTest, GradientCheck_LayerMode);

  int layer_rows_;
  int layer_cols_;

  float epsilon_;  // Small number used for numerical stability.
  int num_samples_;  // Number of minibatch samples seen in the last Forward pass.

  // Internal parameters and their gradients:
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

  // For computing the global mean and variance, needed for inference:
  Phase phase_;
  int phase_sub_id_;

  DeviceMatrix global_mean_;
  DeviceMatrix global_mean_rep_minus_;
  int global_num_samples_;
  DeviceMatrix global_variance_;

  DeviceMatrix global_gamma_;
  DeviceMatrix global_beta_;

};

#endif
