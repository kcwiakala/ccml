#ifndef CCML_ACTIVATION_RELU_HPP
#define CCML_ACTIVATION_RELU_HPP

namespace ccml {
namespace activation {

struct Relu
{
  static double fn(double x) 
  {
    return (x < 0) ? 0 : x;
  }

  static double df(double y)
  {
    return (y > 0) ? 1 : 0;
  }
};

struct LeakingRelu
{
  static double fn(double x) 
  {
    return (x < 0) ? (0.1 * x) : x;
  }

  static double df(double y)
  {
    return (y > 0) ? 1 : -0.1;
  }
};

} // namespace activation
} // namespace ccml

#endif