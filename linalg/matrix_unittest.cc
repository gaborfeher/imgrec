#include "gtest/gtest.h"

#include "linalg/device_matrix.h"

TEST(SmallMatrixTest, HostDeviceTransfer) {
  DeviceMatrix a(2, 2, (float[]){1, 6, 7, 42});
  EXPECT_EQ((std::vector<float> {1, 6, 7, 42}), a.GetVector());
}

TEST(SmallMatrixTest, Add) {
  DeviceMatrix a(2, 2, (float[]){5, 2, 3, 4});
  DeviceMatrix b(2, 2, (float[]){1, 1, 2, 2});
  DeviceMatrix c(a.Add(b));
  EXPECT_EQ((std::vector<float> {6, 3, 5, 6}), c.GetVector());
  EXPECT_EQ(2, c.rows());
  EXPECT_EQ(2, c.cols());
}

TEST(SmallMatrixTest, Transpose) {
  DeviceMatrix a(2, 3, (float[]){1, 2, 3, 4, 5, 6});
  DeviceMatrix at(a.T());
  EXPECT_EQ(
      (std::vector<float> {1, 4, 2, 5, 3, 6}),
      at.GetVector());
  EXPECT_EQ(3, at.rows());
  EXPECT_EQ(2, at.cols());
}

TEST(SmallMatrixTest, Multiply) {
  DeviceMatrix a(2, 3, (float[]){1, 2, 3, 4, 5, 6});
  DeviceMatrix at(a.Multiply(2));
  EXPECT_EQ(
      (std::vector<float> {2, 4, 6, 8, 10, 12}),
      at.GetVector());
  EXPECT_EQ(2, at.rows());
  EXPECT_EQ(3, at.cols());
}
