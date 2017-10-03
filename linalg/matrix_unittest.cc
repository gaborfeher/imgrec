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

TEST(SmallMatrixTest, Rot180) {
  DeviceMatrix a(2, 3, 2, (float[]){
      1, 2, 3,
      4, 5, 6,

      -0.5, 1, 0,
      -0.5, 1, 0
  });
  DeviceMatrix ar(a.Rot180());
  EXPECT_EQ(
      (std::vector<float> {
          6, 5, 4,
          3, 2, 1,

          0, 1, -0.5,
          0, 1, -0.5
      }),
      ar.GetVector());
  EXPECT_EQ(3, ar.rows());
  EXPECT_EQ(2, ar.cols());
  EXPECT_EQ(2, ar.depth());
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

TEST(SmallMatrixTest, Padding) {
  DeviceMatrix a(3, 4, 2, (float[]) {
      1, 1, 2, 2,
      3, 3, 4, 4,
      5, 5, 6, 6,

      1.1, 1.1, 2.2, 2.2,
      3.3, 3.3, 4.4, 4.4,
      5.5, 5.5, 6.6, 6.6});
  EXPECT_EQ(3, a.rows());
  EXPECT_EQ(4, a.cols());
  EXPECT_EQ(2, a.depth());
  DeviceMatrix ap(a.AddPadding(2));
  EXPECT_EQ(7, ap.rows());
  EXPECT_EQ(8, ap.cols());
  EXPECT_EQ(2, ap.depth());
  EXPECT_EQ(
      (std::vector<float> {
          0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 1, 1, 2, 2, 0, 0,
          0, 0, 3, 3, 4, 4, 0, 0,
          0, 0, 5, 5, 6, 6, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0,

          0, 0,   0,   0,   0,   0, 0, 0,
          0, 0,   0,   0,   0,   0, 0, 0,
          0, 0, 1.1, 1.1, 2.2, 2.2, 0, 0,
          0, 0, 3.3, 3.3, 4.4, 4.4, 0, 0,
          0, 0, 5.5, 5.5, 6.6, 6.6, 0, 0,
          0, 0,   0,   0,   0,   0, 0, 0,
          0, 0,   0,   0,   0,   0, 0, 0,
      }),
      ap.GetVector());
}

TEST(SmallMatrixTest, Convolution) {
  // A 3x4 matrix with 3 "color channels"
  DeviceMatrix a(3, 4, 3, (float[]) {
      1, 1, 2, 2,
      3, 3, 4, 4,
      5, 5, 6, 6,

      1.1, 1.1, 2.2, 2.2,
      3.3, 3.3, 4.4, 4.4,
      5.5, 5.5, 6.6, 6.6,

      1, 1, 1, 1,
      1, 1, 1, 1,
      1, 1, 1, 1});
  EXPECT_EQ(3, a.rows());
  EXPECT_EQ(4, a.cols());
  EXPECT_EQ(3, a.depth());
  // Two 2x3x3 filters in a matrix:
  DeviceMatrix c(2, 3, 3 + 3, (float[]) {
    // Filter1:
    1, 1, 1,
    1, 1, 1,

    1, 1, 1,
    1, 1, 1,

    1, 1, 1,
    1, 1, 1,

    // Filter2:
    1, 0.5, 0,
    0, 1, 0.5,

    -1, -1, -1,
    0, 0, 0,

    0, 0, 0,
    2, 2, 2,
  });

  DeviceMatrix ac(a.Convolution(c, 1));
  EXPECT_EQ(2, ac.rows());
  EXPECT_EQ(2, ac.cols());
  EXPECT_EQ(2, ac.depth());
  std::vector<float> expected_vector {
      // Result of the first filter:
      14 + 15.4 + 6, 16 + 17.6 + 6,
      26 + 28.6 + 6, 28 + 30.8 + 6,
      // Result of the second filter:
      6.5 - 4.4 + 6, 8 - 5.5 + 6,
      12.5 - 11 + 6, 14 - 12.1 + 6
  };
  std::vector<float> computed_vector = ac.GetVector();
  EXPECT_EQ(expected_vector.size(), computed_vector.size());
  for (size_t i = 0; i < expected_vector.size(); ++i) {
    EXPECT_FLOAT_EQ(expected_vector[i], computed_vector[i]);
  }
}
