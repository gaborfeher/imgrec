#ifndef _BATCH_NORMALIZATION_LAYER_H_
#define _BATCH_NORMALIZATION_LAYER_H_

#include "cnn/layer.h"

#include "gtest/gtest_prod.h"

class DeviceMatrix;
class Random;

// https://arxiv.org/pdf/1502.03167.pdf
class BatchNormalizationLayer : public Layer {
 public:
  BatchNormalizationLayer(int num_neurons, bool convolutional);
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
  float epsilon_;

  // Input shape and mode description:

  // true= input samples are layers, false= input samples are columns
  bool convolutional_;

  // convolutional_ == true -> number of layers in a sample
  // convolutional_ == false -> number of rows in a sample
  int num_neurons_;

  // ...
  int num_layers_per_sample_;

  int num_samples_;

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
