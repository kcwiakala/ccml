#include <cmath>
#include <transfer/Relu.hpp>

namespace ccml {
namespace transfer {

Relu::Relu(): TransferFunction("relu")
{
}

value_t Relu::apply(value_t x) const
{
  return std::max(x, 0.0);
}

value_t Relu::deriverate(value_t y) const
{
  return (y > 0) ? 1 : 0;
}

} // namespace transfer
} // namespace ccml