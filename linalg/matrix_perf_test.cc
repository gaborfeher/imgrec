#include "gtest/gtest.h"

#include <sstream>
#include <utility>

#include "linalg/matrix.h"
#include "linalg/matrix_test_util.h"

TEST(MatrixPerfTest, Convolution_Fw_NoPadding) {
  // Batch of 400 images, 32 incoming, 32 outgoing layers.
  Matrix a(32, 32, 400 * 32);
  Matrix b(5, 5, 32 * 32);
  Matrix c = Matrix::Convolution(32, a, true, b, true);
}

TEST(MatrixPerfTest, Convolution_Fw) {
  // Batch of 400 images, 32 incoming, 32 outgoing layers.
  Matrix a(32, 32, 400 * 32);
  Matrix b(5, 5, 32 * 32);
  Matrix c = Matrix::Convolution(32, a, true, 2, 2, b, true, 0, 0, 0, 0);
  EXPECT_EQ(32, c.rows());
  EXPECT_EQ(32, c.cols());
  EXPECT_EQ(400 * 32, c.depth());
}

TEST(MatrixPerfTest, Convolution_Bw1) {
  // Batch of 400 images, 32 incoming, 32 outgoing layers.
  Matrix a(32, 32, 400 * 32);
  Matrix b(5, 5, 32 * 32);
  Matrix c = Matrix::Convolution(32, a, true, 4, 4, b, false, 0, 0, 2, 2);
  EXPECT_EQ(32, c.rows());
  EXPECT_EQ(32, c.cols());
  EXPECT_EQ(400 * 32, c.depth());
}

TEST(MatrixPerfTest, Convolution_Bw2) {
  // Batch of 400 images, 32 incoming, 32 outgoing layers.
  Matrix a(32, 32, 400 * 32);
  Matrix b(32, 32, 400 * 32);
  Matrix c = Matrix::Convolution(400, a, false, 4, 4, b, false, 2, 2, 0, 0);
  EXPECT_EQ(5, c.rows());
  EXPECT_EQ(5, c.cols());
  EXPECT_EQ(32 * 32, c.depth());
}
