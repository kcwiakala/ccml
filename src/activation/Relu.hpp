#ifndef CCML_ACTIVATION_RELU_HPP
#define CCML_ACTIVATION_RELU_HPP

#include <activation/detail/Implementation.hpp>

namespace ccml {
namespace activation {

struct Relu: public detail::Implementation<Relu>
{
  static double fn(double x) 
  {
    return (x < 0) ? 0 : x;
  }

  static double df(double y)
  {
    return 1;
  }
};

} // namespace activation
} // namespace ccml

#endif