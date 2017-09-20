#include "DeviceMatrix.h"

#include "HostMatrix.h"

#include <cuda_runtime.h>

#include <iostream>

DeviceMatrix::DeviceMatrix(const HostMatrix& src) :
    rows_(src.rows_),
    cols_(src.cols_),
    size_(src.size_) {
  cudaMalloc(&data_, size_ * sizeof(float)); 
  cudaMemcpy(data_, src.data_.get(), size_ * sizeof(float), cudaMemcpyHostToDevice);
}

DeviceMatrix::~DeviceMatrix() {
  cudaFree(data_);
}
