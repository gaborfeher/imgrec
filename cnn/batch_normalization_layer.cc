#include "cnn/batch_normalization_layer.h"

#include <cassert>

#include "linalg/matrix.h"

BatchNormalizationLayer::BatchNormalizationLayer(int num_neurons, bool layered) :
    BiasLikeLayer(num_neurons, layered),
    epsilon_(0.0001) {
  if (layered) {
    beta_ = Matrix(1, 1, num_neurons);
    gamma_ = Matrix(1, 1, num_neurons);
    global_mean_ = Matrix(1, 1, num_neurons);
    global_variance_ = Matrix(1, 1, num_neurons);
  } else {
    beta_ = Matrix(num_neurons, 1, 1);
    gamma_ = Matrix(num_neurons, 1, 1);
    global_mean_ = Matrix(num_neurons, 1, 1);
    global_variance_ = Matrix(num_neurons, 1, 1);
  }
}

void BatchNormalizationLayer::Initialize(Random*) {
  gamma_.Fill(1.0);
  beta_.Fill(0.0);
}

void BatchNormalizationLayer::Forward(const Matrix& input) {
  input_ = input;

  if (layered_) {
    assert(input.depth() % num_neurons_ == 0);
    // Each layer is considered rows*cols sample of the same
    // variable, because they have to be normalized jointly.
    num_samples_ = input.depth() / num_neurons_ * input.rows() * input.cols();
  } else {
    assert(input.rows() == num_neurons_);
    num_samples_ = input.cols();
  }

  switch (phase_) {
    case NONE:
      break;
    case PRE_TRAIN_PHASE:
      break;
    case POST_TRAIN_PHASE: {
      // Two passes are needed to get global variance:
      assert(phase_sub_id_ == 0 || phase_sub_id_ == 1);
      if (phase_sub_id_ == 0) {
        Matrix sum = input.Sum(layered_, num_neurons_);
        global_mean_ = global_mean_.Add(sum);
        global_num_samples_ += num_samples_;
      } else if (phase_sub_id_ == 1) {
        Matrix local = input
            .Add(global_mean_negative_repeated_)
            .Map(::matrix_mappers::Square())
            .Sum(layered_, num_neurons_);
        global_variance_ = global_variance_.Add(local);
      }
      // break;  // fall through!
    }
    case TRAIN_PHASE: {
      mean_ = input
          .Sum(layered_, num_neurons_)
          .Multiply(1.0 / num_samples_);
      shifted_ = input.Add(
          mean_
              .Multiply(-1.0)
              .Repeat(layered_, input.rows(), input.cols(), input.depth()));
      variance_ = shifted_
          .Map(::matrix_mappers::Square())
          .Sum(layered_, num_neurons_)
          .Multiply(1.0 / num_samples_);
      variance_e_ = variance_.AddConst(epsilon_);
      sqrt_variance_e_ = variance_e_.Map(::matrix_mappers::Sqrt());
      normalized_ = shifted_.ElementwiseDivide(
          sqrt_variance_e_.Repeat(layered_, input.rows(), input.cols(), input.depth()));
      output_ = normalized_
          .ElementwiseMultiply(gamma_.Repeat(layered_, input.rows(), input.cols(), input.depth()))
          .Add(beta_.Repeat(layered_, input.rows(), input.cols(), input.depth()));
      break;
    }

    case INFER_PHASE: {
      Matrix multiplier = global_multiplier_
          .Repeat(layered_, input_.rows(), input_.cols(), input_.depth());
      Matrix shift = global_shift_
          .Repeat(layered_, input_.rows(), input_.cols(), input_.depth());
      output_ = input_
          .ElementwiseMultiply(multiplier)
          .Add(shift);
      break;
    }
  }

}

void BatchNormalizationLayer::Backward(const Matrix& output_gradient) {

  Matrix normalized_grad = output_gradient
      .ElementwiseMultiply(gamma_.Repeat(layered_, input_.rows(), input_.cols(), input_.depth()));
  Matrix variance_grad = normalized_grad
      .ElementwiseMultiply(shifted_)
      .ElementwiseMultiply(variance_e_.Pow(-1.5).Repeat(layered_, input_.rows(), input_.cols(), input_.depth()))
      .Sum(layered_, num_neurons_)
      .Multiply(-0.5);
  Matrix normalized_grad_over_sqrt_variance_e =
      normalized_grad.ElementwiseDivide(sqrt_variance_e_.Repeat(layered_, input_.rows(), input_.cols(), input_.depth()));
  Matrix mean_grad_part1 =
      normalized_grad_over_sqrt_variance_e
          .Sum(layered_, num_neurons_)
          .Multiply(-1.0);
  Matrix mean_grad_part2 = variance_grad
      .ElementwiseMultiply(shifted_.Sum(layered_, num_neurons_))
      .Multiply(-2.0 / num_samples_);
  Matrix mean_grad = mean_grad_part1.Add(mean_grad_part2);

  Matrix input_grad_part1 = normalized_grad_over_sqrt_variance_e;
  Matrix input_grad_part2 = variance_grad
      .Repeat(layered_, input_.rows(), input_.cols(), input_.depth())
      .ElementwiseMultiply(shifted_)
      .Multiply(2.0 / num_samples_);
  Matrix input_grad_part3 = mean_grad
      .Multiply(1.0 / num_samples_)
      .Repeat(layered_, input_.rows(), input_.cols(), input_.depth());

  input_gradient_ = input_grad_part1
      .Add(input_grad_part2)
      .Add(input_grad_part3);
  gamma_gradient_ = output_gradient
      .ElementwiseMultiply(normalized_)
      .Sum(layered_, num_neurons_);
  beta_gradient_ = output_gradient.Sum(layered_, num_neurons_);

}

void BatchNormalizationLayer::ApplyGradient(float learn_rate) {
  beta_.Add(beta_gradient_.Multiply(-learn_rate));
  gamma_.Add(gamma_gradient_.Multiply(-learn_rate));
}

bool BatchNormalizationLayer::BeginPhase(Phase phase, int phase_sub_id) {
  phase_ = phase;
  phase_sub_id_ = phase_sub_id;
  if (phase == POST_TRAIN_PHASE) {
    if (phase_sub_id_ == 0) {
      global_num_samples_ = 0;
      global_mean_.Fill(0.0f);
      global_variance_.Fill(0.0f);
      return true;
    }
    if (phase_sub_id_ == 1) {
      return true;
    }
  }
  return false;
}

void BatchNormalizationLayer::EndPhase(Phase phase, int phase_sub_id) {
  assert(phase_sub_id == phase_sub_id_);
  assert(phase == phase_);
  if (phase_ == POST_TRAIN_PHASE) {
    if (phase_sub_id_ == 0) {
      global_mean_ = global_mean_.Multiply(1.0 / global_num_samples_);
      global_mean_negative_repeated_ = global_mean_
          .Repeat(layered_, input_.rows(), input_.cols(), input_.depth())
          .Multiply(-1.0);
    }
    if (phase_sub_id_ == 1) {
      global_variance_ = global_variance_
          .Multiply(1.0 / global_num_samples_);

      Matrix global_variance_sqrt_e_ =
          global_variance_
              .AddConst(epsilon_)
              .Map(::matrix_mappers::Sqrt());
      global_multiplier_ =
          gamma_.ElementwiseDivide(global_variance_sqrt_e_);
      global_shift_ = beta_
          .Add(
              gamma_
                  .ElementwiseMultiply(global_mean_)
                  .Multiply(-1.0)
                  .ElementwiseDivide(global_variance_sqrt_e_));
    }
  }
  phase_ = NONE;
}
