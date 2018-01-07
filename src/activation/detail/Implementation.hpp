
#include <Activation.hpp>

namespace ccml {
namespace activation {
namespace detail {

template<typename Derived>
class Implementation: public ::ccml::Activation
{
protected:
  Implementation(): Activation(&Derived::fn, &Derived::df) {}
};

} // namespace detail
} // namespace activation
} // namespace ccml