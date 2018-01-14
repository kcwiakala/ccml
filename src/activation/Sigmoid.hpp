#ifndef CCML_ACTIVATION_SIGMOID_HPP
#define CCML_ACTIVATION_SIGMOID_HPP

#include <cmath>

namespace ccml {
namespace activation {

struct Sigmoid
{
  static double fn(double x) 
  {
    return 1 / (1 + std::exp(-x));
  }

  static double df(double y)
  {
    return y * (1 - y);
  }
};

} // namespace activation
} // namespace ccml

#endif