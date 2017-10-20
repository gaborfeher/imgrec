#include "cnn/batch_normalization_layer.h"

#include "linalg/device_matrix.h"

void BatchNormalizationLayer::Forward(const DeviceMatrix&) {
}

void BatchNormalizationLayer::Backward(const DeviceMatrix& output_gradient) {
}


bool BatchNormalizationLayer::BeginTrainingPhase(TrainingPhase phase) {
  return true;
}

void BatchNormalizationLayer::EndTrainingPhase(TrainingPhase phase) {
}
