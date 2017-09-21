#include <iostream>

#include <cuda_runtime.h>

#include "DeviceMatrix.h"
#include "HostMatrix.h"

/*
__global__ void VecAdd(float* A, float* B, float* C) {
  int i = threadIdx.x;
  C[i] = A[i] + B[i];
}
*/

int main() {
  HostMatrix a_host(2, 2, (float[]){5, 2, 3, 4});
  a_host.Print();

  HostMatrix b_host(2, 2, (float[]){1, 1, 2, 2});
  b_host.Print();

  DeviceMatrix a(a_host);
  DeviceMatrix b(b_host);
  DeviceMatrix c(a.Add(b));

  HostMatrix c_host = HostMatrix(c);
  c_host.Print();

  std::cout << "hello, world!" << std::endl;
  return 0;
}
