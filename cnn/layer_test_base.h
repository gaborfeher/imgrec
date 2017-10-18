#ifndef _CNN_LAYER_TEST_BASE_H_
#define _CNN_LAYER_TEST_BASE_H_

#include <functional>

class DeviceMatrix;

// Numerically estimates the gradient of function |runner| at the
// point of |x0|. The result will have the same shape as |x0|.
DeviceMatrix ComputeNumericGradients(
    const DeviceMatrix& x0,
    std::function< float (const DeviceMatrix&) > runner);

#endif  // _CNN_LAYER_TEST_BASE_H_
