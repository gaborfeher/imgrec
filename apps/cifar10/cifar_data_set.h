#ifndef _APPS_CIFAR10_CIFAR_DATA_SET_H_
#define _APPS_CIFAR10_CIFAR_DATA_SET_H_

#include <string>
#include <vector>

#include "infra/data_set.h"

class CifarDataSet : public InMemoryDataSet {
 public:
  CifarDataSet(
      const std::vector<std::string>& file_names,
      int batch_size);

 private:
  int img_size_;
  int images_per_file_;
  int num_classes_;

  void ReadImage(std::ifstream* input, std::vector<float>* x, std::vector<float>* y) const;
};

#endif  // _APPS_CIFAR10_CIFAR_DATA_SET_H_
