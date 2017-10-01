#include <iostream>

#include "linalg/device_matrix.h"

int main() {
  DeviceMatrix a(2, 2, (float[]){5, 2, 3, 4});
  a.Print();

  DeviceMatrix b(2, 2, (float[]){1, 1, 2, 2});
  b.Print();

  DeviceMatrix c(a.Add(b));

  c.Print();

  std::cout << "hello, world!" << std::endl;
  return 0;
}
