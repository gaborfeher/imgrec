#include "infra/data_set.h"

#include <cassert>

#include "linalg/matrix.h"

InMemoryDataSet::InMemoryDataSet(int minibatch_size) :
    num_batches_(0),
    minibatch_size_(minibatch_size) {}

InMemoryDataSet::InMemoryDataSet(
  int minibatch_size,
  const Matrix& x,
  const Matrix& y) :
    num_batches_(0),
    minibatch_size_(minibatch_size) {
  AddBatch(x, y);
}

Matrix InMemoryDataSet::GetBatchInput(int batch) const {
  return x_[batch];
}

Matrix InMemoryDataSet::GetBatchOutput(int batch) const {
  return y_[batch];
}

int InMemoryDataSet::NumBatches() const {
  return num_batches_;
}

int InMemoryDataSet::MiniBatchSize() const {
  return minibatch_size_;
}

void InMemoryDataSet::AddBatch(const Matrix& x, const Matrix& y) {
  num_batches_++;
  x_.push_back(x);
  y_.push_back(y);
}
