#include "infra/data_set.h"

#include <cassert>

#include "linalg/device_matrix.h"

InMemoryDataSet::InMemoryDataSet(int minibatch_size) :
    num_batches_(0),
    minibatch_size_(minibatch_size) {}

InMemoryDataSet::InMemoryDataSet(
  int minibatch_size,
  const DeviceMatrix& x,
  const DeviceMatrix& y) :
    num_batches_(0),
    minibatch_size_(minibatch_size) {
  AddBatch(x, y);
}

DeviceMatrix InMemoryDataSet::GetBatchInput(int batch) const {
  return x_[batch];
}

DeviceMatrix InMemoryDataSet::GetBatchOutput(int batch) const {
  return y_[batch];
}

int InMemoryDataSet::NumBatches() const {
  return num_batches_;
}

int InMemoryDataSet::MiniBatchSize() const {
  return minibatch_size_;
}

void InMemoryDataSet::AddBatch(const DeviceMatrix& x, const DeviceMatrix& y) {
  num_batches_++;
  x_.push_back(x);
  y_.push_back(y);
}
