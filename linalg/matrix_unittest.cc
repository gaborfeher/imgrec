#include "gtest/gtest.h"

#include <iostream>

#include "HostMatrix.h"
#include "DeviceMatrix.h"

TEST(MatrixTest, AddSmall) {
  HostMatrix a_host(2, 2, (float[]){5, 2, 3, 4});
  HostMatrix b_host(2, 2, (float[]){1, 1, 2, 2});
  DeviceMatrix a(a_host);
  DeviceMatrix b(b_host);
  DeviceMatrix c(a.Add(b));
  HostMatrix c_host = HostMatrix(c);
  EXPECT_EQ((std::vector<float> {6, 3, 5, 6}), c_host.GetVector());
}
