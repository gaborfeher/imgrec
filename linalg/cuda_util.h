#ifndef _LINALG_CUDA_UTIL_H_
#define _LINALG_CUDA_UTIL_H_

#include <cassert>

#define CUDA_CALL(fn) do { \
  cudaError_t code = (fn); \
  if (code != cudaSuccess) { \
    gpuAssert(code, __FILE__, __LINE__); \
  } \
} while(0)

#define CURAND_CALL(x) do { if ((x) != CURAND_STATUS_SUCCESS) { \
  fprintf(stderr, "CURAND error at %s:%d\n",__FILE__,__LINE__); \
  exit(1); \
}} while(0)

#define CUDA_ERR_CHECK(ans) do { gpuAssert((ans), __FILE__, __LINE__); } while(0)
inline void gpuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    assert(code != cudaSuccess);
    exit(1);
  }
}

#define CUDA_ASYNC_CHECK() CUDA_ERR_CHECK(cudaPeekAtLastError())

#endif  // _LINALG_CUDA_UTIL_H_
