#ifndef _INFRA_DATA_SET_H_
#define _INFRA_DATA_SET_H_

#include <vector>

#include "linalg/matrix.h"

class DataSet {
 public:
  DataSet() {}
  virtual ~DataSet() {}
  virtual int NumBatches() const = 0;
  virtual int MiniBatchSize() const = 0;
  virtual Matrix GetBatchInput(int batch) const = 0;
  virtual Matrix GetBatchOutput(int batch) const = 0;

  // Prevent copy and assignment.
  DataSet(const DataSet&) = delete;
  DataSet& operator=(const DataSet&) = delete;
};

class InMemoryDataSet : public DataSet {
 public:
  explicit InMemoryDataSet(int minibatch_size);
  InMemoryDataSet(
      int minibatch_size,
      const Matrix& x,
      const Matrix& y);

  virtual int NumBatches() const;
  virtual int MiniBatchSize() const;
  virtual Matrix GetBatchInput(int batch) const;
  virtual Matrix GetBatchOutput(int batch) const;
  void AddBatch(const Matrix& x, const Matrix& y);

 private:
  int num_batches_;
  int minibatch_size_;
  std::vector<Matrix> x_;
  std::vector<Matrix> y_;
};

#endif  // _INFRA_DATA_SET_H_
