#include "gtest/gtest.h"

#include <sstream>
#include <utility>

#include "linalg/matrix.h"
#include "linalg/matrix_test_util.h"

TEST(MatrixPerfTest, Convolution1) {
  // Batch of 400 images, 32 incoming, 32 outgoing layers.
  Matrix a(32, 32, 400 * 32);
  Matrix b(5, 5, 32 * 32);
  Matrix c = Matrix::Convolution(32, a, true, b, true);
}

TEST(MatrixPerfTest, Convolution2) {
  // Batch of 400 images, 32 incoming, 32 outgoing layers.
  Matrix a(32, 32, 400 * 32);
  Matrix b(5, 5, 32 * 32);
  Matrix c = Matrix::Convolution(32, a, 2, 2, true, b, true, 0, 0, 0, 0);
}
