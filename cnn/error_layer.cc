#include "cnn/error_layer.h"

ErrorLayer::ErrorLayer() {}

float ErrorLayer::GetError() const {
  return error_;
}
