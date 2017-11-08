#include <algorithm>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <memory>

#include "cereal/archives/portable_binary.hpp"
#include "cereal/types/memory.hpp"

#include "apps/cifar10/cifar_data_set.h"
#include "cnn/bias_layer.h"
#include "cnn/batch_normalization_layer.h"
#include "cnn/convolutional_layer.h"
#include "cnn/layer_stack.h"
#include "cnn/fully_connected_layer.h"
#include "cnn/input_image_normalization_layer.h"
#include "cnn/inverted_dropout_layer.h"
#include "cnn/nonlinearity_layer.h"
#include "cnn/pooling_layer.h"
#include "cnn/reshape_layer.h"
#include "cnn/softmax_error_layer.h"
#include "infra/logger.h"
#include "infra/trainer.h"
#include "linalg/matrix.h"

void ForceCerealDynamicInit() {
  // Don't call this. It just makes sure that all layer classes
  // are linked in this binary so that cereal can deserialize any
  // kinds of layers here. (btw, cereal has some macros to do this
  // in a prettier way)
  BiasLayer(0, false);
  BatchNormalizationLayer(0, false);
  ConvolutionalLayer(0, 0, 0, 0, 0);
  FullyConnectedLayer(0, 0);
  InputImageNormalizationLayer(0, 0, 0);
  InvertedDropoutLayer(0, false, 0, nullptr);
  PoolingLayer(0, 0);
  ReshapeLayer(0, 0, 0);
  SoftmaxErrorLayer();
  NonlinearityLayer(std::make_pair(nullptr, nullptr));
}

std::shared_ptr<LayerStack> LoadModel(const std::string& fname) {
  std::ifstream is(fname, std::ios::in | std::ios::in);
  cereal::PortableBinaryInputArchive input(is);
  std::shared_ptr<LayerStack> model;
  input(model);
  return model;
}

Matrix LoadImage(const std::string& fname) {
  std::ifstream is(fname, std::ios::in | std::ios::in);
  // Very simple bmp-loading code. We assume 3x8 bit RGB depth
  // and we don't care about row padding because size is 32x32.

  #pragma pack(push)
  #pragma pack(1)
  struct {
    unsigned short int magic;
    unsigned int file_size_bytes;
    unsigned short int reserved1, reserved2;
    unsigned int offset;
  } header;
  struct {
    unsigned int header_size;
    int width, height;
    unsigned short int planes;
    unsigned short int bits_per_pixel;
    unsigned int compression;
    unsigned int imagesize;
    int xresolution, yresolution;
    unsigned int ncolours;
    unsigned int importantcolours;
  } info_header;
  #pragma pack(pop)

  is.read((char*)(&header), sizeof(header));
  is.read((char*)(&info_header), sizeof(info_header));
  assert(header.magic == 0x4D42);
  assert(info_header.width == 32);
  assert(info_header.height == 32);
  assert(info_header.bits_per_pixel == 24);
  assert(info_header.compression == 0);

  is.seekg(header.offset);


  std::vector<float> data;
  Matrix result(32, 32, 3);
  for (int row = 31; row >= 0; --row) {
    for (int col = 0; col < 32; ++col) {
      unsigned char blue = is.get();
      unsigned char green = is.get();
      unsigned char red = is.get();
      result.SetValue(row, col, 0, static_cast<float>(red) / 255.0f);
      result.SetValue(row, col, 1, static_cast<float>(green) / 255.0f);
      result.SetValue(row, col, 2, static_cast<float>(blue) / 255.0f);
    }
  }
  return result;
}

std::vector<std::pair<float, std::string>> MakeNamedScores(Matrix scores) {
  std::vector<std::pair<float, std::string>> result;
  std::ifstream is("apps/cifar10/downloaded_deps/cifar-10-batches-bin/batches.meta.txt");
  for (int i = 0; i < 10; ++i) {
    char buffer[1024];
    is.getline(buffer, 1024);
    result.push_back(std::make_pair(
        scores.GetValue(i, 0, 0),
        buffer));
  }
  return result;
}

void PrintScores(Matrix scores) {
  std::vector<std::pair<float, std::string>> named_scores = MakeNamedScores(scores);
  std::sort(named_scores.begin(), named_scores.end());
  std::reverse(named_scores.begin(), named_scores.end());
  for (int i = 0; i < 10; ++i) {
    std::cout
        << std::fixed
        << std::setw(11) << named_scores[i].second << ": "
        << std::setw(8) << std::setprecision(4) << named_scores[i].first << std::endl;
  }
}

void Test1() {
  // Compares an image from the data set and a bmp image.
  CifarDataSet ds({ "./apps/cifar10/downloaded_deps/cifar-10-batches-bin/data_batch_5.bin" }, 1);
  Matrix img1 = ds.GetBatchInput(0);
  Matrix img2 = LoadImage("./apps/cifar10/downloaded_deps/image0.bmp");
  img1.AssertEquals(img2);
  std::cout << "IMG1 OK" << std::endl;
}

void Test2() {
  // This is how I learned about signed / unsigned char conversions.
  unsigned char ch = 0;
  for (int i = 0; i < 256; i++) {
    std::cout
        << ch << " "
        << static_cast<int>(ch) << " "
        << static_cast<int>(static_cast<char>(ch)) << " "
        << static_cast<int>(static_cast<unsigned char>(static_cast<char>(ch))) << " "
        << std::endl;
    ch++;
  }
}

int main(int args, char* argv[]) {
  // Test1();
  // Test2();
  // return 0;

  assert(args == 3);
  const std::string model_file = argv[1];
  const std::string data_file = argv[2];

  std::shared_ptr<LayerStack> model = LoadModel(model_file);
  if (data_file.rfind(".bmp") == data_file.size() - 4) {
    Matrix image = LoadImage(argv[2]);
    Matrix dummy(1, 1, 1);
    dummy.Fill(0.0f);
    model->GetLayer<SoftmaxErrorLayer>(-1)->SetExpectedValue(dummy);
    model->BeginPhase(Layer::INFER_PHASE, 0);
    model->Forward(image);
    Matrix scores = model->GetLayer<Layer>(-2)->output();
    PrintScores(scores);
  } else if (data_file.rfind(".bin") == data_file.size() - 4) {
    CifarDataSet ds({ data_file }, 400);
    Trainer trainer(model, std::make_shared<Logger>(1));
    float err, acc;
    trainer.Evaluate(ds, &err, &acc);
  } else {
    assert(false);
  }

  return 0;
}
