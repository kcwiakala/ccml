#ifndef CCML_ACTIVATION_HEAVISIDE_HPP
#define CCML_ACTIVATION_HEAVISIDE_HPP

namespace ccml {
namespace activation {

struct Heaviside
{
  static double fn(double x) 
  {
    return (x < 0) ? 0 : 1;
  }

  static double df(double y)
  {
    return 0;
  }
};

} // namespace activation
} // namespace ccml

#endif