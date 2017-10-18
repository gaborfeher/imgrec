#include "cnn/layer_test_base.h"

#include <limits>

#include "gtest/gtest.h"

#include "linalg/device_matrix.h"

DeviceMatrix ComputeNumericGradients(
    const DeviceMatrix& x0,
    std::function< float (const DeviceMatrix&) > runner
) {

  DeviceMatrix result(x0.rows(), x0.cols(), x0.depth());

  float delta = 0.001f;  // I am not super-happy that this is a carefully-tuned value to make all the test pass.
  for (int k = 0; k < x0.depth(); k++) {
    for (int i = 0; i < x0.rows(); i++) {
      for (int j = 0; j < x0.cols(); j++) {
        DeviceMatrix x1(x0.DeepCopy());
        x1.SetValue(i, j, k, x0.GetValue(i, j, k) + delta);
        float error1 = runner(x1);
        x1.SetValue(i, j, k, x0.GetValue(i, j, k) - delta);
        float error2 = runner(x1);

        result.SetValue(i, j, k, (error1 - error2) / (2 * delta) );
      }
    }
  }

  return result;
}
