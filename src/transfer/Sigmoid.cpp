#include <cmath>
#include <transfer/Sigmoid.hpp>

namespace ccml {
namespace transfer {

Sigmoid::Sigmoid(): TransferFunction("sigmoid")
{
}

value_t Sigmoid::apply(value_t x) const
{
  return 1 / (1 + std::exp(-x));
}

value_t Sigmoid::deriverate(value_t y) const
{
  return y * (1 - y);
}

} // namespace transfer
} // namespace ccml