#ifndef CCML_ACTIVATION_SIGMOID_HPP
#define CCML_ACTIVATION_SIGMOID_HPP

#include <cmath>

#include <activation/detail/Implementation.hpp>

namespace ccml {
namespace activation {

struct Sigmoid: public detail::Implementation<Sigmoid>
{
  static double fn(double x) 
  {
    return 1 / (1 + std::exp(-x));
  }

  static double df(double y)
  {
    return y / (1 - y);
  }
};

} // namespace activation
} // namespace ccml

#endif