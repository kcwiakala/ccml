#include <algorithm>
#include <cmath>
#include <sstream>

#include <Transfer.hpp>

namespace ccml {
namespace transfer {

const Transfer& heaviside()
{
  static const Transfer transfer("heaviside", 
    [](value_t x) {
      return (x > 0.0) ? 1.0 : 0.0;
    },
    [](value_t x) {
      return 0;
    },
    [](value_t y) {
      return 0;
    });
  return transfer;
}

const Transfer& identity()
{
  static const Transfer transfer("identity", 
    [](value_t x) {
      return x;
    },
    [](value_t x) {
      return 1;
    },
    [](value_t y) {
      return 1;
    });
  return transfer;
}

const Transfer& sigmoid()
{
  static auto operation = [](value_t x) {
    return 1 / (1 + std::exp(-x));
  };
  static auto derivativeFromY = [](value_t y) {
    return y * (1 - y);
  };
  static auto derivativeFromX = [=](value_t x) {
    return derivativeFromY(operation(x));
  };
  static const Transfer transfer("sigmoid", operation, derivativeFromX, derivativeFromY);
  return transfer;
}

const Transfer& relu()
{
  static auto operation = [=](value_t x) {
    return std::max(x, 0.0);
  };
  static auto derivative = [](value_t x) {
    return (x > 0.0) ? 1.0 : 0.0;
  };
  static const Transfer transfer("relu", operation, derivative, derivative);
  return transfer;
}

Transfer leakingRelu(value_t leakingRate)
{
  auto operation = [=](value_t x) {
    return (x > 0) ? x : (leakingRate * x);
  };
  auto derivative = [=](value_t x) {
    return (x > 0) ? 1 : leakingRate;
  };
  std::stringstream name;
  name << "leakingRelu(" << leakingRate << ")";
  return std::move(Transfer(name.str(), operation, derivative, derivative));
}

} // namespace transfer
} // namespace ccml