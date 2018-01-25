#include <transfer/Heaviside.hpp>

namespace ccml {
namespace transfer {

Heaviside::Heaviside(): Transfer("heaviside")
{
}

value_t Heaviside::apply(value_t x) const
{
  return (x > 0) ? 1 : 0;
}

value_t Heaviside::deriverate(value_t y) const
{
  return 0;
}

} // namespace transfer
} // namespace ccml