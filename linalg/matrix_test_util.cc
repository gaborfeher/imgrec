#include "linalg/matrix_test_util.h"

#include <limits>

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
    EXPECT_NEAR(av[i], bv[i], absolute_diff) << "(i= " << i << ")";
    if (percentage_diff >= 0.0f) {
      float diff = std::abs(av[i] - bv[i]);
      if (diff < std::numeric_limits<float>::epsilon() ||
          av[i] < std::numeric_limits<float>::epsilon() ||
          bv[i] < std::numeric_limits<float>::epsilon()) {
        // In this case we don't care about percentages, because
        // of float precision woodoo issues.
        continue;
      }
      float magnitude = ((std::abs(av[i]) + std::abs(bv[i])) / 2.0);
      if (magnitude > 0.0f) {
        EXPECT_LT(
            100.0 * diff / magnitude,
            percentage_diff)
            << "(i=" << i
            << " a= " << av[i]
            << " b= " << bv[i]
            << ")";
      }
    }
  }
}

void ExpectMatrixEquals(
    const DeviceMatrix& a,
    const DeviceMatrix& b) {
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
    EXPECT_FLOAT_EQ(av[i], bv[i]) << "(i= " << i << ")";
  }
}
