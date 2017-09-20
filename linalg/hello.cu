#include <iostream>

#include <cuda_runtime.h>

#include "DeviceMatrix.h"
#include "HostMatrix.h"


__global__ void VecAdd(float* A, float* B, float* C) {
  int i = threadIdx.x;
  C[i] = A[i] + B[i];
}

int main() {
  HostMatrix ha(2, 2, (float[]){5, 2, 3, 4});
  ha.Print();
  DeviceMatrix da(ha);
  HostMatrix ha2(da);
  ha2.Print();


  std::cout << "hello, world!" << std::endl;


  float h_A[] = { 1, 2, 3, 4, 5 };
  float h_B[] = { 1, 2, 1, 2, 1 };
  float h_C[5];

  int N = 5;
  size_t size = N * sizeof(float);

  float* d_A;
  cudaMalloc(&d_A, size);
  float* d_B;
  cudaMalloc(&d_B, size);
  float* d_C;
  cudaMalloc(&d_C, size);

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  VecAdd<<<1, 5>>>(d_A, d_B, d_C);

  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; ++i) {
    std::cout << h_C[i] << " ";
  }
  std::cout << std::endl;

  return 0;
}
