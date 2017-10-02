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

TEST(SmallMatrixTest, ElementwiseMultiply) {
  DeviceMatrix a(2, 2, (float[]){5, 2, 3, 4});
  DeviceMatrix b(2, 2, (float[]){1, 1, 2, 2});
  DeviceMatrix c(a.ElementwiseMultiply(b));
  EXPECT_EQ((std::vector<float> {5, 2, 6, 8}), c.GetVector());
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
  DeviceMatrix am(a.Multiply(2));
  EXPECT_EQ(
      (std::vector<float> {2, 4, 6, 8, 10, 12}),
      am.GetVector());
  EXPECT_EQ(2, am.rows());
  EXPECT_EQ(3, am.cols());
}

TEST(SmallMatrixTest, DotProduct) {
  DeviceMatrix a(2, 3, (float[]){
      1, 2, 3,
      4, 5, 6});
  DeviceMatrix b(3, 4, (float[]){
      1,  2,  3,  4,
      5,  6,  7,  8,
      9, 10, 11, 12});

  DeviceMatrix c(a.Dot(b));
  EXPECT_EQ(
      (std::vector<float> {
          38, 44,  50,  56,
          83, 98, 113, 128
      }),
      c.GetVector());
  EXPECT_EQ(2, c.rows());
  EXPECT_EQ(4, c.cols());
}

TEST(SmallMatrixTest, Sigmoid) {
  DeviceMatrix a(1, 2, (float[]){0, 1});
  DeviceMatrix as(a.ApplySigmoid());
  EXPECT_FLOAT_EQ(0.5, as.GetVector()[0]);
  EXPECT_NEAR(0.73105, as.GetVector()[1], 0.00001);
  EXPECT_EQ(1, as.rows());
  EXPECT_EQ(2, as.cols());
}

TEST(SmallMatrixTest, SigmoidGradients) {
  DeviceMatrix a(1, 2, (float[]){0, 1});
  DeviceMatrix as(a.ApplySigmoidGradients());
  EXPECT_FLOAT_EQ(0.25, as.GetVector()[0]);
  EXPECT_NEAR(0.19661, as.GetVector()[1], 0.00001);
  EXPECT_EQ(1, as.rows());
  EXPECT_EQ(2, as.cols());
}

TEST(SmallMatrixTest, L2) {
  DeviceMatrix a(2, 2, (float[]){1, 1, 2, 0.5});
  DeviceMatrix al(a.L2());
  EXPECT_FLOAT_EQ(2.5, al.GetVector()[0]);
  EXPECT_EQ(1, al.rows());
  EXPECT_EQ(1, al.cols());
}

TEST(SmallMatrixTest, Fill) {
  DeviceMatrix a(2, 2, (float[]){1, 1, 2, 0.5});
  a.Fill(4.2);
  EXPECT_EQ(
      (std::vector<float> {
        4.2, 4.2,
        4.2, 4.2
      }),
      a.GetVector());
  EXPECT_EQ(2, a.rows());
  EXPECT_EQ(2, a.cols());
}

TEST(SmallMatrixTest, ZeroInit) {
  DeviceMatrix a(2, 2);
  EXPECT_EQ(
      (std::vector<float> {
          0.0, 0.0,
          0.0, 0.0
      }),
      a.GetVector());
  EXPECT_EQ(2, a.rows());
  EXPECT_EQ(2, a.cols());
}

