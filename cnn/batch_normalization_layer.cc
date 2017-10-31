#include "cnn/batch_normalization_layer.h"

#include <cassert>
#include <iostream>

BatchNormalizationLayer::BatchNormalizationLayer(int num_neurons, bool layered) :
    BiasLikeLayer(num_neurons, layered),
    epsilon_(0.0001) {
  if (layered) {
    beta_ = MatrixParam(1, 1, num_neurons);
    gamma_ = MatrixParam(1, 1, num_neurons);
    global_mean_ = Matrix(1, 1, num_neurons);
    global_variance_ = Matrix(1, 1, num_neurons);
    global_multiplier_ = Matrix(1, 1, num_neurons);
    global_shift_ = Matrix(1, 1, num_neurons);
  } else {
    beta_ = MatrixParam(num_neurons, 1, 1);
    gamma_ = MatrixParam(num_neurons, 1, 1);
    global_mean_ = Matrix(num_neurons, 1, 1);
    global_variance_ = Matrix(num_neurons, 1, 1);
    global_multiplier_ = Matrix(num_neurons, 1, 1);
    global_shift_ = Matrix(num_neurons, 1, 1);
  }
}

void BatchNormalizationLayer::Print() const {
  std::cout << "Batch Normalization Layer" << std::endl;
  std::cout << " Multiplier:" << std::endl;
  global_multiplier_.Print();
  std::cout << " Shift:" << std::endl;
  global_shift_.Print();
}

void BatchNormalizationLayer::Initialize(Random*) {
  gamma_.value.Fill(1.0);
  beta_.value.Fill(0.0);
  global_multiplier_.Fill(1.0);
  global_shift_.Fill(0.0);
}

void BatchNormalizationLayer::Forward(const Matrix& input) {
  input_ = input;  // (This is only needed in one place for dimensions.)

  if (layered_) {
    assert(input.depth() % num_neurons_ == 0);
    // Each layer is considered rows*cols sample of the same
    // variable, because they have to be normalized jointly.
    num_samples_ = input.depth() / num_neurons_ * input.rows() * input.cols();
  } else {
    assert(input.rows() == num_neurons_);
    num_samples_ = input.cols();
  }

  switch (phase()) {
    case NONE:
      assert(false);
      break;
    case POST_TRAIN_PHASE: {
      // Two passes are needed to get global variance. (Other layers
      // may use more passes.)
      if (phase_sub_id() == 0) {
        Matrix sum = input.Sum(layered_, num_neurons_);
        global_mean_ = global_mean_.Add(sum);
        global_num_samples_ += num_samples_;
      } else if (phase_sub_id() == 1) {
        Matrix local = input
            .Add(global_mean_negative_repeated_)
            .Map1(::matrix_mappers::Square())
            .Sum(layered_, num_neurons_);
        global_variance_ = global_variance_.Add(local);
      }
      // break;  // fall through
    }
    case PRE_TRAIN_PHASE:  // fall through
    case TRAIN_PHASE: {
      mean_ = input
          .Sum(layered_, num_neurons_)
          .Multiply(1.0 / num_samples_);
      shifted_ = input.Add(
          mean_
              .Multiply(-1.0)
              .Repeat(layered_, input));
      variance_ = shifted_
          .Map1(::matrix_mappers::Square())
          .Sum(layered_, num_neurons_)
          .Multiply(1.0 / num_samples_);
      variance_plus_e_ = variance_.AddConst(epsilon_);
      sqrt_variance_plus_e_repeated_ = variance_plus_e_
          .Map1(::matrix_mappers::Sqrt())
          .Repeat(layered_, input.rows(), input.cols(), input.depth());
      normalized_ = shifted_.ElementwiseDivide(sqrt_variance_plus_e_repeated_);
      output_ = normalized_
          .ElementwiseMultiply(gamma_.value.Repeat(layered_, input))
          .Add(beta_.value.Repeat(layered_, input));
      break;
    }

    case INFER_PHASE: {
      Matrix multiplier = global_multiplier_
          .Repeat(layered_, input);
      Matrix shift = global_shift_
          .Repeat(layered_, input);
      output_ = input
          .ElementwiseMultiply(multiplier)
          .Add(shift);
      break;
    }
  }

}

void BatchNormalizationLayer::Backward(const Matrix& output_gradient) {
  Matrix normalized_grad = output_gradient
      .ElementwiseMultiply(gamma_.value.Repeat(layered_, output_gradient));
  Matrix variance_grad = normalized_grad
      .ElementwiseMultiply(shifted_)
      .ElementwiseMultiply(variance_plus_e_.Pow(-1.5).Repeat(layered_, output_gradient))
      .Sum(layered_, num_neurons_)
      .Multiply(-0.5);
  Matrix normalized_grad_over_sqrt_variance_e =
      normalized_grad.ElementwiseDivide(sqrt_variance_plus_e_repeated_);
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
      .Repeat(layered_, output_gradient)
      .ElementwiseMultiply(shifted_)
      .Multiply(2.0 / num_samples_);
  Matrix input_grad_part3 = mean_grad
      .Multiply(1.0 / num_samples_)
      .Repeat(layered_, output_gradient);

  input_gradient_ = input_grad_part1
      .Add(input_grad_part2)
      .Add(input_grad_part3);
  gamma_.gradient = output_gradient
      .ElementwiseMultiply(normalized_)
      .Sum(layered_, num_neurons_);
  beta_.gradient = output_gradient
      .Sum(layered_, num_neurons_);
}

void BatchNormalizationLayer::ApplyGradient(const GradientInfo& info) {
  GradientInfo copy = info;
  copy.lambda = 0.0f;  // no regularization
  beta_.ApplyGradient(copy);
  gamma_.ApplyGradient(copy);
}

bool BatchNormalizationLayer::OnBeginPhase() {
  if (phase() == POST_TRAIN_PHASE) {
    if (phase_sub_id() == 0) {
      global_num_samples_ = 0;
      global_mean_.Fill(0.0f);
      global_variance_.Fill(0.0f);
      return true;
    }
    if (phase_sub_id() == 1) {
      return true;
    }
  }
  return false;
}

void BatchNormalizationLayer::OnEndPhase() {
  if (phase() == POST_TRAIN_PHASE) {
    if (phase_sub_id() == 0) {
      global_mean_ = global_mean_.Multiply(1.0 / global_num_samples_);
      global_mean_negative_repeated_ = global_mean_
          .Repeat(layered_, input_)
          .Multiply(-1.0);
    }
    if (phase_sub_id() == 1) {
      global_variance_ = global_variance_
          .Multiply(1.0 / global_num_samples_);

      Matrix global_variance_sqrt_e_ =
          global_variance_
              .AddConst(epsilon_)
              .Map1(::matrix_mappers::Sqrt());
      global_multiplier_ =
          gamma_.value.ElementwiseDivide(global_variance_sqrt_e_);
      global_shift_ = beta_.value
          .Add(
              gamma_.value
                  .ElementwiseMultiply(global_mean_)
                  .Multiply(-1.0)
                  .ElementwiseDivide(global_variance_sqrt_e_));
    }
  }
}

int BatchNormalizationLayer::NumParameters() const {
  return beta_.NumParameters() + gamma_.NumParameters();
}
