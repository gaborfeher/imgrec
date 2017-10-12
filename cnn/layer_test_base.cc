#include "cnn/layer_test_base.h"

#include "gtest/gtest.h"

#include "linalg/device_matrix.h"

void ExpectMatrixEquals(
    const DeviceMatrix& a,
    const DeviceMatrix& b,
    float absolute_diff,
    float percentage_diff) {
  EXPECT_EQ(a.rows(), b.rows());
  EXPECT_EQ(a.cols(), b.cols());
  EXPECT_EQ(a.depth(), b.depth());
  std::vector<float> av = a.GetVector();
  std::vector<float> bv = b.GetVector();
  EXPECT_EQ(av.size(), bv.size());
  if (av.size() != bv.size()) {
    return;
  }
  for (size_t i = 0; i < av.size(); ++i) {
    EXPECT_NEAR(av[i], bv[i], absolute_diff);
    if (percentage_diff >= 0.0f) {
      float magnitude = ((std::abs(av[i]) + std::abs(bv[i])) / 2.0);
      if (magnitude > 0.0f) {
        EXPECT_LT(
            100.0f * std::abs(av[i] - bv[i]) / magnitude,
            percentage_diff)
            << "(i=" << i
            << " a= " << av[i]
            << " b= " << bv[i]
            << ")";
      }
    }
  }
}

DeviceMatrix ComputeNumericGradients(
    const DeviceMatrix& x0,
    std::function< float (const DeviceMatrix&) > runner
) {

  DeviceMatrix result(x0.rows(), x0.cols(), x0.depth());

  float error0 = runner(x0);
  float delta = 0.002f;  // I am not supper-happy that this is a carefully-tuned value to make all the test pass.
  for (int k = 0; k < x0.depth(); k++) {
    for (int i = 0; i < x0.rows(); i++) {
      for (int j = 0; j < x0.cols(); j++) {
        DeviceMatrix x1(x0.DeepCopy());
        x1.SetValue(i, j, k, x0.GetValue(i, j, k) + delta);
        float error1 = runner(x1);
        result.SetValue(i, j, k, (error1 - error0) / delta );
      }
    }
  }

  return result;
}
