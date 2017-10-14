#include "gtest/gtest.h"

#include "linalg/device_matrix.h"

TEST(SmallMatrixTest, HostDeviceTransfer) {
  DeviceMatrix a(2, 2, 1, (float[]){1, 6, 7, 42});
  EXPECT_EQ((std::vector<float> {1, 6, 7, 42}), a.GetVector());
}

TEST(SmallMatrixTest, Add) {
  DeviceMatrix a(2, 2, 1, (float[]){5, 2, 3, 4});
  DeviceMatrix b(2, 2, 1, (float[]){1, 1, 2, 2});
  DeviceMatrix c(a.Add(b));
  EXPECT_EQ((std::vector<float> {6, 3, 5, 6}), c.GetVector());
  EXPECT_EQ(2, c.rows());
  EXPECT_EQ(2, c.cols());
}

TEST(SmallMatrixTest, ElementwiseMultiply) {
  DeviceMatrix a(2, 2, 1, (float[]){5, 2, 3, 4});
  DeviceMatrix b(2, 2, 1, (float[]){1, 1, 2, 2});
  DeviceMatrix c(a.ElementwiseMultiply(b));
  EXPECT_EQ((std::vector<float> {5, 2, 6, 8}), c.GetVector());
  EXPECT_EQ(2, c.rows());
  EXPECT_EQ(2, c.cols());
}

TEST(SmallMatrixTest, Transpose) {
  DeviceMatrix a(2, 3, 1, (float[]){1, 2, 3, 4, 5, 6});
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
  EXPECT_EQ(2, ar.rows());
  EXPECT_EQ(3, ar.cols());
  EXPECT_EQ(2, ar.depth());
}

TEST(SmallMatrixTest, Multiply) {
  DeviceMatrix a(2, 3, 1, (float[]){1, 2, 3, 4, 5, 6});
  DeviceMatrix am(a.Multiply(2));
  EXPECT_EQ(
      (std::vector<float> {2, 4, 6, 8, 10, 12}),
      am.GetVector());
  EXPECT_EQ(2, am.rows());
  EXPECT_EQ(3, am.cols());
}

TEST(SmallMatrixTest, DotProduct) {
  DeviceMatrix a(2, 3, 1, (float[]){
      1, 2, 3,
      4, 5, 6});
  DeviceMatrix b(3, 4, 1, (float[]){
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
  DeviceMatrix a(1, 2, 1, (float[]){0, 1});
  DeviceMatrix as(a.Map(matrix_mappers::Sigmoid()));
  EXPECT_FLOAT_EQ(0.5, as.GetVector()[0]);
  EXPECT_NEAR(0.73105, as.GetVector()[1], 0.00001);
  EXPECT_EQ(1, as.rows());
  EXPECT_EQ(2, as.cols());
}

TEST(SmallMatrixTest, SigmoidGradient) {
  DeviceMatrix a(1, 2, 1, (float[]){0, 1});
  DeviceMatrix as(a.Map(matrix_mappers::SigmoidGradient()));
  EXPECT_FLOAT_EQ(0.25, as.GetVector()[0]);
  EXPECT_NEAR(0.19661, as.GetVector()[1], 0.00001);
  EXPECT_EQ(1, as.rows());
  EXPECT_EQ(2, as.cols());
}

TEST(SmallMatrixTest, L2) {
  DeviceMatrix a(2, 2, 1, (float[]){1, 1, 2, 0.5});
  EXPECT_FLOAT_EQ(2.5, a.L2());
}

TEST(SmallMatrixTest, Softmax1) {
  // from http://cs231n.github.io/linear-classify/#softmax
  DeviceMatrix wx(3, 1, 1, (float[]) {-2.85, 0.86, 0.28});
  DeviceMatrix y(1, 1, 1, (float[]) {2.0});
  EXPECT_NEAR(1.04, wx.Softmax(y), 0.0002);
}

TEST(SmallMatrixTest, Softmax2) {
  // from http://cs231n.github.io/linear-classify/#softmax
  DeviceMatrix wx(3, 2, 1, (float[]) {
      -2.85, 2.0,
      0.86, -1.0,
      0.28, 1.4,
  });
  DeviceMatrix y(1, 2, 1, (float[]) {2.0, 1.0});
  EXPECT_NEAR((1.0402 + 3.4691) / 2.0, wx.Softmax(y), 0.0002);
}

TEST(SmallMatrixTest, Fill) {
  DeviceMatrix a(2, 2, 1, (float[]){1, 1, 2, 0.5});
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
  DeviceMatrix a(2, 2, 1);
  EXPECT_EQ(
      (std::vector<float> {
          0.0, 0.0,
          0.0, 0.0
      }),
      a.GetVector());
  EXPECT_EQ(2, a.rows());
  EXPECT_EQ(2, a.cols());
}

TEST(SmallMatrixTest, AddPadding) {
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
  DeviceMatrix ap(a.AddConstRow(4.2f));
  EXPECT_EQ(4, ap.rows());
  EXPECT_EQ(4, ap.cols());
  EXPECT_EQ(2, ap.depth());
  EXPECT_EQ(
      (std::vector<float> {
          1, 1, 2, 2,
          3, 3, 4, 4,
          5, 5, 6, 6,
          4.2, 4.2, 4.2, 4.2,

          1.1, 1.1, 2.2, 2.2,
          3.3, 3.3, 4.4, 4.4,
          5.5, 5.5, 6.6, 6.6,
          4.2, 4.2, 4.2, 4.2,
      }),
      ap.GetVector());
}

TEST(SmallMatrixTest, AddConstRow) {
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
  DeviceMatrix ap(a.AddPadding(2, 1));
  EXPECT_EQ(7, ap.rows());
  EXPECT_EQ(6, ap.cols());
  EXPECT_EQ(2, ap.depth());
  EXPECT_EQ(
      (std::vector<float> {
          0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0,
          0, 1, 1, 2, 2, 0,
          0, 3, 3, 4, 4, 0,
          0, 5, 5, 6, 6, 0,
          0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0,

          0,   0,   0,   0,   0, 0,
          0,   0,   0,   0,   0, 0,
          0, 1.1, 1.1, 2.2, 2.2, 0,
          0, 3.3, 3.3, 4.4, 4.4, 0,
          0, 5.5, 5.5, 6.6, 6.6, 0,
          0,   0,   0,   0,   0, 0,
          0,   0,   0,   0,   0, 0,
      }),
      ap.GetVector());
}

TEST(SmallMatrixTest, ReduceSize) {
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
  DeviceMatrix ar(a.ReduceSize(2, 3, 2));
  EXPECT_EQ(2, ar.rows());
  EXPECT_EQ(3, ar.cols());
  EXPECT_EQ(2, ar.depth());
  EXPECT_EQ(
      (std::vector<float> {
          1, 1, 2,
          3, 3, 4,

          1.1, 1.1, 2.2,
          3.3, 3.3, 4.4,
      }),
      ar.GetVector());
}

TEST(SmallMatrixTest, Convolution) {
  // Two 3x4 images with 3 "color channels" each:
  DeviceMatrix a(3, 4, 3 * 2, (float[]) {
      // Image1, layer1:
      1, 1, 2, 2,
      3, 3, 4, 4,
      5, 5, 6, 6,
      // Image1, layer2:
      1.1, 1.1, 2.2, 2.2,
      3.3, 3.3, 4.4, 4.4,
      5.5, 5.5, 6.6, 6.6,
      // Image1, layer3:
      1, 1, 1, 1,
      1, 1, 1, 1,
      1, 1, 1, 1,
      // Image2, layer1:
      0, 1, 2, 2,
      3, 3, 4, 4,
      5, 5, 6, 6,
      // Image2, layer2:
      1.1, 1.1, 2.2, 2.2,
      3.3, 3.3, 4.4, 4.4,
      5.5, 5.5, 6.6, 6.6,
      // Image2, layer3:
      1, 1, 1, 1,
      1, 1, 1, 1,
      0, 1, 1, 1
  });
  EXPECT_EQ(3, a.rows());
  EXPECT_EQ(4, a.cols());
  EXPECT_EQ(6, a.depth());
  // Two 2x3x3 filters in a matrix:
  DeviceMatrix c(2, 3, 3 * 2, (float[]) {
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

  DeviceMatrix ac(a.Convolution(c, 3, 1));
  EXPECT_EQ(2, ac.rows());
  EXPECT_EQ(2, ac.cols());
  EXPECT_EQ(4, ac.depth());
  std::vector<float> expected_vector {
      // Result of the 1st filter on 1st image:
      14 + 15.4 + 6, 16 + 17.6 + 6,
      26 + 28.6 + 6, 28 + 30.8 + 6,
      // Result of the 2nd filter on 1st image:
      6.5 - 4.4 + 6, 8 - 5.5 + 6,
      12.5 - 11 + 6, 14 - 12.1 + 6,
      // Result of the 1st filter on 2nd image:
      14 + 15.4 + 6 - 1, 16 + 17.6 + 6,
      26 + 28.6 + 6 - 1, 28 + 30.8 + 6,
      // Result of the 2nd filter on 2nd image:
      6.5 - 4.4 + 6 - 1, 8 - 5.5 + 6,
      12.5 - 11 + 6 - 2, 14 - 12.1 + 6,
  };
  std::vector<float> computed_vector = ac.GetVector();
  EXPECT_EQ(expected_vector.size(), computed_vector.size());
  for (size_t i = 0; i < expected_vector.size(); ++i) {
    EXPECT_FLOAT_EQ(expected_vector[i], computed_vector[i]);
  }
}

TEST(SmallMatrixTest, ConvolutionWithBias) {
  // Two 3x4 images with 3 "color channels" each:
  DeviceMatrix a(3, 4, 3 * 2, (float[]) {
      // Image1, layer1:
      1, 1, 2, 2,
      3, 3, 4, 4,
      5, 5, 6, 6,
      // Image1, layer2:
      1.1, 1.1, 2.2, 2.2,
      3.3, 3.3, 4.4, 4.4,
      5.5, 5.5, 6.6, 6.6,
      // Image1, layer3:
      1, 1, 1, 1,
      1, 1, 1, 1,
      1, 1, 1, 1,
      // Image2, layer1:
      0, 1, 2, 2,
      3, 3, 4, 4,
      5, 5, 6, 6,
      // Image2, layer2:
      1.1, 1.1, 2.2, 2.2,
      3.3, 3.3, 4.4, 4.4,
      5.5, 5.5, 6.6, 6.6,
      // Image2, layer3:
      1, 1, 1, 1,
      1, 1, 1, 1,
      0, 1, 1, 1
  });
  EXPECT_EQ(3, a.rows());
  EXPECT_EQ(4, a.cols());
  EXPECT_EQ(6, a.depth());
  // Two 2x3x3 filters in a matrix:
  DeviceMatrix c(2, 3, 3 * 2, (float[]) {
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
  DeviceMatrix bias(1, 1, 2, (float[]) { +1.0f, -1.0f });

  DeviceMatrix ac(a.Convolution(c, 3, 1, bias));
  EXPECT_EQ(2, ac.rows());
  EXPECT_EQ(2, ac.cols());
  EXPECT_EQ(4, ac.depth());
  std::vector<float> expected_vector {
      // Result of the 1st filter on 1st image:
      14 + 15.4 + 6 + 1, 16 + 17.6 + 6 + 1,
      26 + 28.6 + 6 + 1, 28 + 30.8 + 6 + 1,
      // Result of the 2nd filter on 1st image:
      6.5 - 4.4 + 6 - 1, 8 - 5.5 + 6 - 1,
      12.5 - 11 + 6 - 1, 14 - 12.1 + 6 - 1,
      // Result of the 1st filter on 2nd image:
      14 + 15.4 + 6 - 1 + 1, 16 + 17.6 + 6 + 1,
      26 + 28.6 + 6 - 1 + 1, 28 + 30.8 + 6 + 1,
      // Result of the 2nd filter on 2nd image:
      6.5 - 4.4 + 6 - 1 - 1, 8 - 5.5 + 6 - 1,
      12.5 - 11 + 6 - 2 - 1, 14 - 12.1 + 6 - 1,
  };
  std::vector<float> computed_vector = ac.GetVector();
  EXPECT_EQ(expected_vector.size(), computed_vector.size());
  for (size_t i = 0; i < expected_vector.size(); ++i) {
    EXPECT_FLOAT_EQ(expected_vector[i], computed_vector[i]);
  }
}

TEST(SmallMatrixTest, Reshape) {
  DeviceMatrix m(2, 3, 4, (float[]) {
    1, 2, 3,
    4, 5, 6,

    7, 8, 9,
    10, 11, 12,

    13, 14, 15,
    16, 17, 18,

    19, 20, 21,
    22, 23, 24,
  });

  DeviceMatrix r(m.ReshapeToColumns(2));
  EXPECT_EQ(12, r.rows());
  EXPECT_EQ(2, r.cols());
  EXPECT_EQ(1, r.depth());
  EXPECT_EQ(
      (std::vector<float> {
          1, 13,
          2, 14,
          3, 15,
          4, 16,
          5, 17,
          6, 18,
          7, 19,
          8, 20,
          9, 21,
          10, 22,
          11, 23,
          12, 24,
      }),
      r.GetVector());
  DeviceMatrix rr(r.ReshapeFromColumns(2, 3, 2));
  EXPECT_EQ(m.rows(), rr.rows());
  EXPECT_EQ(m.cols(), rr.cols());
  EXPECT_EQ(m.depth(), rr.depth());
  EXPECT_EQ(m.GetVector(), rr.GetVector());
}

TEST(SmallMatrixTest, ReorderLayers) {
  DeviceMatrix m(2, 3, 6, (float[]) {
    // image1, layer1
    1, 2, 3,
    4, 5, 6,
    // image1, layer2
    7, 8, 9,
    10, 11, 12,
    // image1, layer3
    13, 14, 15,
    16, 17, 18,
    // image2, layer1
    19, 20, 21,
    22, 23, 24,
    // image2, layer2
    25, 26, 27,
    28, 29, 30,
    // image2, layer3
    31, 32, 33,
    34, 35, 36,
  });
  DeviceMatrix rl(m.ReorderLayers(3));
  EXPECT_EQ(2, rl.rows());
  EXPECT_EQ(3, rl.cols());
  EXPECT_EQ(6, rl.depth());
  EXPECT_EQ(
      (std::vector<float> {
          // new image1, layer1
          1, 2, 3,
          4, 5, 6,
          // new image1, layer2
          19, 20, 21,
          22, 23, 24,
          // new image2, layer1
          7, 8, 9,
          10, 11, 12,
          // new image2, layer2
          25, 26, 27,
          28, 29, 30,
          // new image3, layer1
          13, 14, 15,
          16, 17, 18,
          // new image3, layer2
          31, 32, 33,
          34, 35, 36,
      }),
      rl.GetVector());
}


TEST(SmallMatrixTest, CopyGetSet) {
  DeviceMatrix a(2, 3, 2, (float[]){
      1, 2, 3,
      4, 5, 6,

      7, 8, 9,
      10, 11, 12,
  });
  DeviceMatrix b(a.DeepCopy());
  EXPECT_EQ(8, a.GetValue(0, 1, 1));
  EXPECT_EQ(8, b.GetValue(0, 1, 1));
  b.SetValue(0, 1, 1, 42);
  EXPECT_EQ(8, a.GetValue(0, 1, 1));
  EXPECT_EQ(42, b.GetValue(0, 1, 1));

  EXPECT_EQ(
      (std::vector<float> {
          1, 2, 3,
          4, 5, 6,

          7, 8, 9,
          10, 11, 12,
      }),
      a.GetVector());
  EXPECT_EQ(
      (std::vector<float> {
          1, 2, 3,
          4, 5, 6,

          7, 42, 9,
          10, 11, 12,
      }),
      b.GetVector());
}
