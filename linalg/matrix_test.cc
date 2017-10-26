#include "gtest/gtest.h"

#include <utility>

#include "linalg/matrix.h"
#include "linalg/matrix_test_util.h"

TEST(SmallMatrixTest, HostDeviceTransfer) {
  Matrix a(2, 2, 1, (float[]){1, 6, 7, 42});
  EXPECT_EQ((std::vector<float> {1, 6, 7, 42}), a.GetVector());
}

TEST(SmallMatrixTest, Add) {
  Matrix a(2, 2, 1, (float[]){5, 2, 3, 4});
  Matrix b(2, 2, 1, (float[]){1, 1, 2, 2});
  Matrix c(a.Add(b));
  EXPECT_EQ((std::vector<float> {6, 3, 5, 6}), c.GetVector());
  EXPECT_EQ(2, c.rows());
  EXPECT_EQ(2, c.cols());
}

TEST(SmallMatrixTest, AddConst) {
  Matrix a(2, 2, 1, (float[]){5, 2, 3, 4});
  Matrix b(a.AddConst(1));
  Matrix b_exp(2, 2, 1, (float[]){6, 3, 4, 5});
  ExpectMatrixEquals(b_exp, b);
}

TEST(SmallMatrixTest, Pow) {
  Matrix a(2, 2, 1, (float[]){5, 2, 3, 4});
  Matrix b(a.Pow(3));
  Matrix b_exp(2, 2, 1, (float[]){125, 8, 27, 64});
  ExpectMatrixEquals(b_exp, b);
}

TEST(SmallMatrixTest, Square) {
  Matrix a(2, 2, 1, (float[]){5, 2, -3, 4});
  Matrix b(a.Map(::matrix_mappers::Square()));
  Matrix b_exp(2, 2, 1, {25, 4, 9, 16});
  ExpectMatrixEquals(b_exp, b);
}

TEST(SmallMatrixTest, Sqrt) {
  Matrix a(2, 2, 1, {25, 4, 9, 16});
  Matrix b(a.Map(::matrix_mappers::Sqrt()));
  Matrix b_exp(2, 2, 1, (float[]){5, 2, 3, 4});
  ExpectMatrixEquals(b_exp, b);
}

TEST(SmallMatrixTest, ElementwiseMultiply) {
  Matrix a(2, 2, 1, (float[]){5, 2, 3, 4});
  Matrix b(2, 2, 1, (float[]){1, 1, 2, 2});
  Matrix c(a.ElementwiseMultiply(b));
  EXPECT_EQ((std::vector<float> {5, 2, 6, 8}), c.GetVector());
  EXPECT_EQ(2, c.rows());
  EXPECT_EQ(2, c.cols());
}

TEST(SmallMatrixTest, ElementwiseDivide) {
  Matrix a(2, 2, 1, (float[]){5, 2, 3, 4});
  Matrix b(2, 2, 1, (float[]){1, 1, 2, 2});
  Matrix c(a.ElementwiseDivide(b));
  Matrix c_exp(2, 2, 1, (float[]) {5, 2, 1.5, 2});
  ExpectMatrixEquals(c_exp, c);
}

TEST(SmallMatrixTest, Transpose) {
  Matrix a(2, 3, 1, (float[]){1, 2, 3, 4, 5, 6});
  Matrix at(a.T());
  EXPECT_EQ(
      (std::vector<float> {1, 4, 2, 5, 3, 6}),
      at.GetVector());
  EXPECT_EQ(3, at.rows());
  EXPECT_EQ(2, at.cols());
}

TEST(SmallMatrixTest, Rot180) {
  Matrix a(2, 3, 2, (float[]){
      1, 2, 3,
      4, 5, 6,

      -0.5, 1, 0,
      -0.5, 1, 0
  });
  Matrix ar(a.Rot180());
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
  Matrix a(2, 3, 1, (float[]){1, 2, 3, 4, 5, 6});
  Matrix am(a.Multiply(2));
  EXPECT_EQ(
      (std::vector<float> {2, 4, 6, 8, 10, 12}),
      am.GetVector());
  EXPECT_EQ(2, am.rows());
  EXPECT_EQ(3, am.cols());
}

TEST(SmallMatrixTest, Divide) {
  Matrix a(2, 3, 1, (float[]){1, 2, 3, 4, 5, 6});
  Matrix am(a.Divide(0.5));
  EXPECT_EQ(
      (std::vector<float> {2, 4, 6, 8, 10, 12}),
      am.GetVector());
  EXPECT_EQ(2, am.rows());
  EXPECT_EQ(3, am.cols());
}

TEST(SmallMatrixTest, DotProduct) {
  Matrix a(2, 3, 1, (float[]){
      1, 2, 3,
      4, 5, 6});
  Matrix b(3, 4, 1, (float[]){
      1,  2,  3,  4,
      5,  6,  7,  8,
      9, 10, 11, 12});

  Matrix c(a.Dot(b));
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
  Matrix a(1, 2, 1, (float[]){0, 1});
  Matrix as(a.Map(matrix_mappers::Sigmoid()));
  EXPECT_FLOAT_EQ(0.5, as.GetVector()[0]);
  EXPECT_NEAR(0.73105, as.GetVector()[1], 0.00001);
  EXPECT_EQ(1, as.rows());
  EXPECT_EQ(2, as.cols());
}

TEST(SmallMatrixTest, SigmoidGradient) {
  Matrix a(1, 2, 1, (float[]){0, 1});
  Matrix as(a.Map(matrix_mappers::SigmoidGradient()));
  EXPECT_FLOAT_EQ(0.25, as.GetVector()[0]);
  EXPECT_NEAR(0.19661, as.GetVector()[1], 0.00001);
  EXPECT_EQ(1, as.rows());
  EXPECT_EQ(2, as.cols());
}

TEST(SmallMatrixTest, L2) {
  Matrix a(2, 2, 1, (float[]){1, 1, 2, 0.5});
  EXPECT_FLOAT_EQ(6.25f, a.L2());
}

TEST(SmallMatrixTest, Softmax1) {
  // from http://cs231n.github.io/linear-classify/#softmax
  Matrix wx(3, 1, 1, (float[]) {-2.85, 0.86, 0.28});
  Matrix y(1, 1, 1, (float[]) {2.0});
  EXPECT_NEAR(1.04, wx.Softmax(y), 0.0002);
}

TEST(SmallMatrixTest, Softmax2) {
  // from http://cs231n.github.io/linear-classify/#softmax
  Matrix wx(3, 2, 1, (float[]) {
      -2.85, 2.0,
      0.86, -1.0,
      0.28, 1.4,
  });
  Matrix y(1, 2, 1, (float[]) {2.0, 1.0});
  EXPECT_NEAR((1.0402 + 3.4691), wx.Softmax(y), 0.0002);
}

TEST(SmallMatrixTest, NumMatches) {
  Matrix wx(3, 4, 1, (float[]) {
      -2.85, 2.0, 1.0, 3.0,
      0.86, 1.4, -1.0, 2.0,
      0.9, 1.4, -2.0, 1.0,
  });
  Matrix y(1, 4, 1, (float[]) {2.0, 1.0, 1.0, 0.0});
  EXPECT_FLOAT_EQ(2.0f, wx.NumMatches(y));
}

TEST(SmallMatrixTest, Fill) {
  Matrix a(2, 2, 1, (float[]){1, 1, 2, 0.5});
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
  Matrix a(2, 2, 1);
  EXPECT_EQ(
      (std::vector<float> {
          0.0, 0.0,
          0.0, 0.0
      }),
      a.GetVector());
  EXPECT_EQ(2, a.rows());
  EXPECT_EQ(2, a.cols());
}

TEST(SmallMatrixTest, AddRemovePadding) {
  Matrix a(3, 4, 2, (float[]) {
      1, 1, 2, 2,
      3, 3, 4, 4,
      5, 5, 6, 6,

      1.1, 1.1, 2.2, 2.2,
      3.3, 3.3, 4.4, 4.4,
      5.5, 5.5, 6.6, 6.6});
  EXPECT_EQ(3, a.rows());
  EXPECT_EQ(4, a.cols());
  EXPECT_EQ(2, a.depth());
  Matrix ap(a.AddPadding(2, 1));
  ExpectMatrixEquals(
      Matrix(7, 6, 2, (float[]) {
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
      ap);
  Matrix arp = ap.RemovePadding(2, 1);
  ExpectMatrixEquals(a, arp);
}

TEST(SmallMatrixTest, Convolution) {
  // Two 3x4 images with 3 "color channels" each:
  Matrix a(3, 4, 3 * 2, (float[]) {
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
  Matrix c(2, 3, 3 * 2, (float[]) {
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

  Matrix ac(a.Convolution(c, 3));
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

TEST(SmallMatrixTest, Reshape) {
  Matrix m(2, 3, 4, (float[]) {
    1, 2, 3,
    4, 5, 6,

    7, 8, 9,
    10, 11, 12,

    13, 14, 15,
    16, 17, 18,

    19, 20, 21,
    22, 23, 24,
  });

  Matrix r(m.ReshapeToColumns(2));
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
  Matrix rr(r.ReshapeFromColumns(2, 3, 2));
  EXPECT_EQ(m.rows(), rr.rows());
  EXPECT_EQ(m.cols(), rr.cols());
  EXPECT_EQ(m.depth(), rr.depth());
  EXPECT_EQ(m.GetVector(), rr.GetVector());
}

TEST(SmallMatrixTest, ReorderLayers) {
  Matrix m(2, 3, 6, (float[]) {
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
  Matrix rl(m.ReorderLayers(3));
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
  Matrix a(2, 3, 2, (float[]){
      1, 2, 3,
      4, 5, 6,

      7, 8, 9,
      10, 11, 12,
  });
  Matrix b(a.DeepCopy());
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

TEST(SmallMatrixTest, Sum_Layers) {
  // Two 3x4 images with 3 "color channels" each:
  Matrix a(3, 4, 6, (float[]) {
      // layer1:
      1, 1, 2, 2,
      3, 3, 4, 4,
      5, 5, 6, 6,
      // layer2:
      1.1, 1.1, 2.2, 2.2,
      3.3, 3.3, 4.4, 4.4,
      5.5, 5.5, 6.6, 6.6,
      // layer3:
      1, 1, 1, 1,
      1, 1, 1, 1,
      1, 1, 1, 1,
      // layer4:
      0, 1, 2, 2,
      3, 3, 4, 4,
      5, 5, 6, 6,
      // layer5:
      1.1, 1.1, 2.2, 2.2,
      3.3, 3.3, 4.4, 4.4,
      5.5, 5.5, 6.6, 6.6,
      // layer6:
      1, 1, 1, 1,
      1, 1, 1, 1,
      0, 1, 1, 1
  });
  Matrix s = a.Sum(true, 3);
  EXPECT_EQ(1, s.rows());
  EXPECT_EQ(1, s.cols());
  EXPECT_EQ(3, s.depth());
  EXPECT_FLOAT_EQ(
      42  /* layer1 */ + 41  /* layer4 */,
      s.GetValue(0, 0, 0));
  EXPECT_FLOAT_EQ(
      46.2 /* layer2 */ + 46.2 /* layer5 */,
      s.GetValue(0, 0, 1));
  EXPECT_FLOAT_EQ(
      12 /* layer3 */ + 11 /* layer5 */,
      s.GetValue(0, 0, 2));
}

TEST(SmallMatrixTest, PerLayerSum) {
  // Two 3x4 images with 3 "color channels" each:
  Matrix a(3, 4, 6, (float[]) {
      // layer1:
      1, 1, 2, 2,
      3, 3, 4, 4,
      5, 5, 6, 6,
      // layer2:
      1.1, 1.1, 2.2, 2.2,
      3.3, 3.3, 4.4, 4.4,
      5.5, 5.5, 6.6, 6.6,
      // layer3:
      1, 1, 1, 1,
      1, 1, 1, 1,
      1, 1, 1, 1,
      // layer4:
      0, 1, 2, 2,
      3, 3, 4, 4,
      5, 5, 6, 6,
      // layer5:
      1.1, 1.1, 2.2, 2.2,
      3.3, 3.3, 4.4, 4.4,
      5.5, 5.5, 6.6, 6.6,
      // layer6:
      1, 1, 1, 1,
      1, 1, 1, 1,
      0, 1, 1, 1
  });
  Matrix s = a.PerLayerSum(3);
  ExpectMatrixEquals(
    Matrix(3, 4, 3, (float[]) {
      // layer1 + layer4:
      1, 2, 4, 4,
      6, 6, 8, 8,
      10, 10, 12, 12,
      // layer2 + layer5:
      2.2, 2.2, 4.4, 4.4,
      6.6, 6.6, 8.8, 8.8,
      11.0, 11.0, 13.2, 13.2,
      // layer3 + layer6:
      2, 2, 2, 2,
      2, 2, 2, 2,
      1, 2, 2, 2,
    }),
    s);
}

TEST(SmallMatrixTest, PerLayerRepeat) {
  Matrix a(2, 3, 2, (float[]) {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9,
    10, 11, 12,
  });
  Matrix s = a.PerLayerRepeat(3);
  ExpectMatrixEquals(
    Matrix(2, 3, 6, (float[]) {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
    }),
    s);
}
TEST(SmallMatrixTest, Sum_Columns) {
  Matrix a(3, 4, 1, (float[]) {
      1, 1, 2, 2,
      3, 3, 4, 4,
      5, 5, 6, 6,
  });
  Matrix s = a.Sum(false, 3);
  Matrix expected(3, 1, 1, (float[]) {
      6,
      14,
      22,
  });
  ExpectMatrixEquals(s, expected);
}

TEST(SmallMatrixTest, Repeat_Layers) {
  Matrix a(1, 1, 2, (float[]) {
      1,
      2,
  });
  Matrix b = a.Repeat(true, 2, 3, 6);
  Matrix expected(2, 3, 6, (float[]) {
      1, 1, 1,
      1, 1, 1,
      2, 2, 2,
      2, 2, 2,
      1, 1, 1,
      1, 1, 1,
      2, 2, 2,
      2, 2, 2,
      1, 1, 1,
      1, 1, 1,
      2, 2, 2,
      2, 2, 2,
  });
  ExpectMatrixEquals(expected, b);
}

TEST(SmallMatrixTest, Repeat_Columns) {
  Matrix a(4, 1, 1, (float[]) {
      4,
      3,
      2,
      1,
  });
  Matrix b = a.Repeat(false, 4, 3, 1);
  Matrix expected(4, 3, 1, (float[]) {
      4, 4, 4,
      3, 3, 3,
      2, 2, 2,
      1, 1, 1,
  });
  ExpectMatrixEquals(expected, b);
}

TEST(SmallMatrixTest, Pooling_PoolingSwitch) {
  Matrix a(4, 6, 2, (float[]) {
      1, 2, 3, 4, 5, 6,
      7, 8, 9, 10, 11, 12,
      4, -2, 3, -2, -1, -5,
      2, 1, -3, -2, -3, -4,

      4, -2, 3, -2, -1, -5,
      2, 1, -3, -2, -3, -4,
      1, 2, 3, 4, 5, 6,
      7, 8, 9, 10, 11, 12,
  });
  std::pair<Matrix, Matrix> res = a.Pooling(2, 3);
  {
    SCOPED_TRACE("res.first");
    ExpectMatrixEquals(
        Matrix(2, 2, 2, (float[]) {
            9, 12,
            4, -1,

            4, -1,
            9, 12,
        }),
        res.first);
  }
  {
    SCOPED_TRACE("res.second");
    ExpectMatrixEquals(
        Matrix(2, 2, 2, (float[]) {
            5, 5,
            0, 1,

            0, 1,
            5, 5,
        }),
        res.second);
  }

  // Test PoolingSwitch
  Matrix b(2, 2, 2, (float[]) {
    1, 2,
    3, 4,
    5, 6,
    7, 8
  });
  Matrix switched = b.PoolingSwitch(res.second, 2, 3);
  {
    SCOPED_TRACE("switched");
    ExpectMatrixEquals(
        Matrix(4, 6, 2, (float[]) {
            0, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 2,
            3, 0, 0, 0, 4, 0,
            0, 0, 0, 0, 0, 0,

            5, 0, 0, 0, 6, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 7, 0, 0, 8,
        }),
        switched);
  }
}

TEST(BigMatrixTest, DotProduct) {
  Matrix a(11, 3, 1);
  a.Fill(1.0f);
  Matrix b(3, 200, 1);
  b.Fill(2.0f);
  Matrix c(a.Dot(b));
  Matrix c_exp(11, 200, 1);
  c_exp.Fill(6.0f);
  ExpectMatrixEquals(c_exp, c);
}

TEST(BigMatrixTest, Fill) {
  Matrix a(100, 100, 5);
  a.Fill(42.0f);
  for (int i = 0; i < 100; ++i) {
    for (int j = 0; j < 100; ++j) {
      for (int k = 0; k < 5; ++k) {
        EXPECT_FLOAT_EQ(42.0f, a.GetValue(i, j, k));
      }
    }
  }
}

TEST(BigMatrixTest, ReLU) {
  Matrix a(100, 100, 5);
  a.Fill(2.0f);
  Matrix b = a.Map(matrix_mappers::ReLU());

  Matrix b_exp(100, 100, 5);
  b_exp.Fill(2.0f);
  ExpectMatrixEquals(b_exp, b);
}
