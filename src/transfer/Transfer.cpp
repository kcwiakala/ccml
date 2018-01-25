#include <algorithm>
#include <cmath>
#include <sstream>

#include <iostream>

#include <unordered_map>

#include <transfer/Transfer.hpp>

namespace ccml {

const std::string& Transfer::name() const noexcept
{
  return _name;
}

void Transfer::apply(array_t& x) const
{
  std::transform(x.cbegin(), x.cend(), x.begin(), [&](value_t xi) {
    return apply(xi);
  });
}

void Transfer::apply(const array_t& x, array_t& y) const
{
  std::transform(x.cbegin(), x.cend(), y.begin(), [&](value_t xi) {
    return apply(xi);
  });
}

void Transfer::deriverate(const array_t& y, array_t& dx) const
{
  if(&y != &dx)
  {
    dx.resize(y.size());
  }
  std::transform(y.cbegin(), y.cend(), dx.begin(), [&](value_t yi) {
    return deriverate(yi);
  });
}

value_t Transfer::apply(value_t x) const
{
  return x;
}

value_t Transfer::deriverate(value_t y) const
{
  return 1;
}

} // namespace ccml