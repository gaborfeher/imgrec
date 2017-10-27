#include "apps/cifar10/cifar_data_set.h"

#include <cassert>
#include <fstream>
#include <iostream>

#include "linalg/matrix.h"

CifarDataSet::CifarDataSet(
    const std::vector<std::string>& file_names,
    int batch_size) :
        InMemoryDataSet(batch_size),
        img_size_(1024 * 3),
        images_per_file_(10000) {
  assert(images_per_file_ % batch_size == 0);
  for (const std::string& file_name: file_names) {
    std::cout << file_name << std::endl;
    std::ifstream input(file_name, std::ios::binary);

    for (int batch_id = 0; batch_id < images_per_file_ / batch_size; ++batch_id) {
      std::vector<float> x;
      std::vector<float> y;
      x.reserve(batch_size * img_size_);
      y.reserve(batch_size);
      for (int i = 0; i < batch_size; ++i) {
        ReadImage(&input, &x, &y);
      }
      AddBatch(
          Matrix(32, 32, 3 * batch_size, x),
          Matrix(1, batch_size, 1, y));
    }

    input.close();
  }
}

void CifarDataSet::ReadImage(
    std::ifstream* input,
    std::vector<float> *x, std::vector<float>* y) const {
  char buffer[img_size_ + 1];
  assert(!input->eof() && input->good());
  input->read(buffer, img_size_ + 1);
  // assert that the read was successful:
  assert(input->eof() || input->good());

  int class_id = static_cast<int>(buffer[0]);
  y->push_back(class_id);

  for (int i = 1; i < img_size_ + 1; ++i) {
    x->push_back(static_cast<float>(buffer[i]) / 256.0f);
  }
  
}
