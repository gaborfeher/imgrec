#include "cnn/layer.h"

#include <cassert>
#include <iostream>

Layer::Layer() :
    phase_(NONE),
    phase_sub_id_(-1) {}

void Layer::Print() const {
  std::cout << Name() << std::endl;
}

bool Layer::BeginPhase(Phase phase, int phase_sub_id) {
  phase_ = phase;
  phase_sub_id_ = phase_sub_id;
  return OnBeginPhase();
}

void Layer::EndPhase(Phase phase, int phase_sub_id) {
  OnEndPhase();
  assert(phase_sub_id == phase_sub_id_);
  assert(phase == phase_);
  phase_ = NONE;
  phase_sub_id_ = -1;
}
