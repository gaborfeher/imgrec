#include "cnn/error_layer.h"

ErrorLayer::ErrorLayer() {}

void ErrorLayer::ApplyGradient(float) {
}

float ErrorLayer::GetError() const {
  return error_;
}
