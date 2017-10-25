#ifndef _LINALG_MATRIX_TEST_UTIL_H_
#define _LINALG_MATRIX_TEST_UTIL_H_

class Matrix;

void ExpectMatrixEquals(
    const Matrix& a,
    const Matrix& b,
    float absolute_diff,
    float percentage_diff);

void ExpectMatrixEquals(
    const Matrix& a,
    const Matrix& b);

#endif  // _LINALG_MATRIX_TEST_UTIL_H_
