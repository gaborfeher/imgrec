#include "gtest/gtest.h"

#include "linalg/host_matrix.h"
#include "linalg/device_matrix.h"

TEST(SmallMatrixTest, HostDeviceTransfer) {
  HostMatrix a_host1(2, 2, (float[]){1, 6, 7, 42});
  DeviceMatrix a_device(a_host1);
  HostMatrix a_host2(a_device);
  EXPECT_EQ((std::vector<float> {1, 6, 7, 42}), a_host1.GetVector());
  EXPECT_EQ((std::vector<float> {1, 6, 7, 42}), a_host2.GetVector());
}

TEST(SmallMatrixTest, Add) {
  HostMatrix a_host(2, 2, (float[]){5, 2, 3, 4});
  HostMatrix b_host(2, 2, (float[]){1, 1, 2, 2});
  DeviceMatrix a(a_host);
  DeviceMatrix b(b_host);
  DeviceMatrix c(a.Add(b));
  HostMatrix c_host = HostMatrix(c);
  EXPECT_EQ((std::vector<float> {6, 3, 5, 6}), c_host.GetVector());
  EXPECT_EQ(2, c_host.rows());
  EXPECT_EQ(2, c_host.cols());
}

TEST(SmallMatrixTest, Transpose) {
  HostMatrix a_host(2, 3, (float[]){1, 2, 3, 4, 5, 6});
  DeviceMatrix a(a_host);
  DeviceMatrix at(a.T());
  HostMatrix at_host(at);
  EXPECT_EQ(
      (std::vector<float> {1, 4, 2, 5, 3, 6}),
      at_host.GetVector());
  EXPECT_EQ(3, at_host.rows());
  EXPECT_EQ(2, at_host.cols());
}
