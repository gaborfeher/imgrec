#include "cnn/batch_normalization_layer.h"

#include <cassert>

#include "linalg/device_matrix.h"

void BatchNormalizationLayer::Forward(const DeviceMatrix& input) {
  input_ = input;

  switch (phase_) {
    case TRAIN_PHASE: {
      float epsilon = 0.0001;

      if (num_layers_per_sample_ > 0) {
        assert(input.depth() % num_layers_per_sample_ == 0);
        num_samples_ = input.depth() / num_layers_per_sample_;
      } else {
        num_samples_ = input.cols();
      }

      mean_ = input
          .Sum(num_layers_per_sample_)
          .Multiply(1.0 / num_samples_);
      shifted_ = input.AddPattern(
          num_layers_per_sample_, mean_.Multiply(-1.0));
      variance_ = shifted_
          .Map(::matrix_mappers::SQUARE)
          .Sum(num_layers_per_sample_)
          .Multiply(1.0 / num_samples_);
      variance_e_ = variance_.Add(epsilon);
      sqrt_variance_e_ = variance_e_
          .Map(::matrix_mappers::SQRT);
      normalized_ = shifted_.ElementwiseDivide(sqrt_variance_e_);
      output_ = normalized_.ElementwiseMultiply(gamma_).Add(beta_);

      break;
    }
    case POST_TRAIN_PHASE:
      // Two passes are needed to get global variance:
      // 1. Compute mean.
      // 2. Compute variance.
      break;
    case INFER_PHASE:
      // Use global variance to infer.
      break;
  }

}

void BatchNormalizationLayer::Backward(const DeviceMatrix& output_gradient) {

  DeviceMatrix normalized_grad = output_gradient.ElementwiseMultiply(gamma_);
  DeviceMatrix variance_grad = normalized_grad
      .ElementwiseMultiply(shifted_)
      .ElementwiseMultiply(variance_e_.Pow(-1.5))
      .Sum(num_layers_per_sample_)
      .Multiply(-0.5);
  DeviceMatrix normalized_grad_over_sqrt_variance_e =
      normalized_grad.ElementwiseDivide(sqrt_variance_e_)
  DeviceMatrix mean_grad_part1 =
      normalized_grad_over_sqrt_variance_e
          .Sum(num_layers_per_sample_)
          .Multiply(-1.0);
  DeviceMatrix mean_grad_part2 = variance_grad
      .ElementwiseMultiply(shifted_.Sum(num_layers_per_sample_))
      .Multiply(-2.0 / num_samples_);
  DeviceMatrix mean_grad = mean_grad_part1.Add(mean_grad_part2);

  DeviceMatrix input_grad_part1 = normalized_grad_over_sqrt_variance_e;
  DeviceMatrix input_grad_part2 = variance_grad
      .ElementwiseMultiply(shifted_.Sum(num_layers_per_sample_) /*???*/)
      .Multiply(2.0 / num_samples_);
  DeviceMatrix input_grad_part3 = mean_grad.Multiply(1.0 / num_samples_);
  input_gradients_ = input_grad_part1
      .Add(input_grad_part2)
      .Add(input_grad_part3);
  gamma_gradient_ = output_gradient
      .ElementwiseMultiply(normalized_)
      .Sum(num_layers_per_sample_);
  beta_gradient_ = output_gradient.Sum(num_layers_per_sample_);

}

void BatchNormalizationLayer::ApplyGradient(float learn_rate) {
  beta_.Add(beta_gradient_.Multiply(-learn_rate));
  gamma_.Add(gamma_gradient_.Multiply(-learn_rate));
}

int BatchNormalizationLayer::BeginPhase(Phase phase) {
  phase_ = phase;
  return phase == POST_TRAIN_PHASE ? 2 : 0;
}

void BatchNormalizationLayer::EndPhase(Phase phase) {
  phase_ = NONE;
}
