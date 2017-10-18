#ifndef _LINALG_MATRIX_TEST_UTIL_H_
#define _LINALG_MATRIX_TEST_UTIL_H_

class DeviceMatrix;

void ExpectMatrixEquals(
    const DeviceMatrix& a,
    const DeviceMatrix& b,
    float absolute_diff,
    float percentage_diff);

void ExpectMatrixEquals(
    const DeviceMatrix& a,
    const DeviceMatrix& b);

#endif  // _LINALG_MATRIX_TEST_UTIL_H_
