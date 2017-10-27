#ifndef _UTIL_RANDOM_H_
#define _UTIL_RANDOM_H_

#include <random>

class Random {
 public:
  Random(int seed) : rnd_(seed) {}

  template <class Distribution>
  int RandInt(Distribution* distribution) {
    return (*distribution)(rnd_);
  }

  template <class Distribution>
  float RandFloat(Distribution* distribution) {
    return (*distribution)(rnd_);
  }

  unsigned long RandLongUnsigned() {
    return rnd_();
  }

 private:
  std::mt19937 rnd_;


};

#endif  // _UTIL_RANDOM_H_
