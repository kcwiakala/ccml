#include <cmath>
#include <transfer/LeakingRelu.hpp>

namespace ccml {
namespace transfer {

LeakingRelu::LeakingRelu(double leakingRate): 
  Transfer("leakingRelu(" + std::to_string(leakingRate) + ")"),
  _leakingRate(leakingRate)
{
}

value_t LeakingRelu::apply(value_t x) const
{
  return (x > 0) ? x : (_leakingRate * x);
}

value_t LeakingRelu::deriverate(value_t y) const
{
  return (y > 0) ? 1 : _leakingRate;
}

} // namespace transfer
} // namespace ccml