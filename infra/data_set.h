#ifndef _INFRA_DATA_SET_H_
#define _INFRA_DATA_SET_H_

#include <vector>

class DeviceMatrix;

class DataSet {
 public:
  DataSet() {}
  virtual ~DataSet() {}
  virtual int NumBatches() const = 0;
  virtual int MiniBatchSize() const = 0;
  virtual DeviceMatrix GetBatchInput(int batch) const = 0;
  virtual DeviceMatrix GetBatchOutput(int batch) const = 0;

  // Prevent copy and assignment.
  DataSet(const DataSet&) = delete;
  DataSet& operator=(const DataSet&) = delete;
};

class InMemoryDataSet : public DataSet {
 public:
  InMemoryDataSet(
      int minibatch_size);
  InMemoryDataSet(
      int minibatch_size,
      const DeviceMatrix& x,
      const DeviceMatrix& y);

  virtual int NumBatches() const;
  virtual int MiniBatchSize() const;
  virtual DeviceMatrix GetBatchInput(int batch) const;
  virtual DeviceMatrix GetBatchOutput(int batch) const;
  void AddBatch(const DeviceMatrix& x, const DeviceMatrix& y);

 private:
  int num_batches_;
  int minibatch_size_;
  std::vector<DeviceMatrix> x_;
  std::vector<DeviceMatrix> y_;
};

#endif  // _INFRA_DATA_SET_H_
