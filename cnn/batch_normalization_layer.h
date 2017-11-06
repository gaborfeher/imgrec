#ifndef _BATCH_NORMALIZATION_LAYER_H_
#define _BATCH_NORMALIZATION_LAYER_H_

#include "cnn/bias_like_layer.h"
#include "cnn/matrix_param.h"
#include "linalg/matrix.h"

#include "gtest/gtest_prod.h"

class Matrix;
class Random;
namespace cereal {
class PortableBinaryOutputArchive;
class PortableBinaryInputArchive;
class access;
}

// https://arxiv.org/pdf/1502.03167.pdf
class BatchNormalizationLayer : public BiasLikeLayer {
 public:
  // See BiasLikeLayer for param docs.
  BatchNormalizationLayer(int num_neurons, bool layered);
  virtual void Print() const;
  virtual void Initialize(Random*);
  virtual void Forward(const Matrix& input);
  virtual void Backward(const Matrix& output_gradient);
  virtual void ApplyGradient(const GradientInfo& info);
  virtual bool OnBeginPhase();
  virtual void OnEndPhase();
  virtual int NumParameters() const;

  // serialization/deserialization
  void save(cereal::PortableBinaryOutputArchive& ar) const;
  void load(cereal::PortableBinaryInputArchive& ar);
 private:
  BatchNormalizationLayer() : BiasLikeLayer(0, false) {}  // for cereal
  friend class cereal::access;

  FRIEND_TEST(BatchNormalizationLayerTest, ForwardNormalization_ColumnMode);
  FRIEND_TEST(BatchNormalizationLayerTest, ForwardBetaGamma_ColumnMode);
  FRIEND_TEST(BatchNormalizationLayerTest, Forward_LayerMode);
  FRIEND_TEST(BatchNormalizationLayerTest, GradientCheck_ColumnMode);
  FRIEND_TEST(BatchNormalizationLayerTest, GradientCheck_LayerMode);
  FRIEND_TEST(BatchNormalizationLayerTest, GlobalSum_ColumnMode);
  FRIEND_TEST(BatchNormalizationLayerTest, GlobalSum_LayerMode);
  FRIEND_TEST(BatchNormalizationLayerTest, Infer_ColumnMode);
  FRIEND_TEST(BatchNormalizationLayerTest, Infer_LayerMode);

  float epsilon_;  // Small number used for numerical stability.
  int num_samples_;  // Number of minibatch samples seen in the last Forward pass.

  // Internal parameters and their gradients:
  MatrixParam beta_;
  MatrixParam gamma_;

  // Intermediate values shared between Forward and Backward.
  Matrix mean_;
  Matrix shifted_;
  Matrix variance_;
  Matrix variance_plus_e_;
  Matrix sqrt_variance_plus_e_repeated_;
  Matrix normalized_;

  // For computing the global mean and variance, needed for inference:
  Matrix global_mean_;
  Matrix global_mean_negative_repeated_;
  int global_num_samples_;
  Matrix global_variance_;

  Matrix global_multiplier_;
  Matrix global_shift_;
};

#endif
