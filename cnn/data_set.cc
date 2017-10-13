#include "cnn/data_set.h"

#include "linalg/device_matrix.h"

InMemoryDataSet::InMemoryDataSet() {}

InMemoryDataSet::InMemoryDataSet(DeviceMatrix x, DeviceMatrix y) :
    num_batches_(1),
    x_{ x },
    y_{ y }
{}

DeviceMatrix InMemoryDataSet::GetBatchInput(int batch) const {
  return x_[batch];
}

DeviceMatrix InMemoryDataSet::GetBatchOutput(int batch) const {
  return y_[batch];
}

int InMemoryDataSet::NumBatches() const {
  return num_batches_;
}


void InMemoryDataSet::AddBatch(const DeviceMatrix& x, const DeviceMatrix& y) {
  num_batches_++;
  x_.push_back(x);
  y_.push_back(y);
}
