#include "gtest/gtest.h"

#include <iostream>

#include "linalg/host_matrix.h"
#include "linalg/device_matrix.h"

TEST(MatrixTest, HostDeviceTransfer) {
  HostMatrix a_host1(2, 2, (float[]){1, 6, 7, 42});
  DeviceMatrix a_device(a_host1);
  HostMatrix a_host2(a_device);
  std::cout << "construction done" << std::endl;
  EXPECT_EQ((std::vector<float> {1, 6, 7, 42}), a_host1.GetVector());
  EXPECT_EQ((std::vector<float> {1, 6, 7, 42}), a_host2.GetVector());
  a_host1.Print();
  a_host2.Print();
}
/*
TEST(MatrixTest, AddSmall) {
  HostMatrix a_host(2, 2, (float[]){5, 2, 3, 4});
  HostMatrix b_host(2, 2, (float[]){1, 1, 2, 2});
  DeviceMatrix a(a_host);
  DeviceMatrix b(b_host);
  DeviceMatrix c(a.Add(b));
  HostMatrix c_host = HostMatrix(c);
  EXPECT_EQ((std::vector<float> {6, 3, 5, 6}), c_host.GetVector());
}
*/
