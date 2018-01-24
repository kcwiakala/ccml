#include <algorithm>
#include <cmath>
#include <transfer/Softmax.hpp>

namespace ccml {
namespace transfer {

Softmax::Softmax(): TransferFunction("softmax")
{
}

void Softmax::apply(const array_t& x, array_t& y) const
{
  y.resize(x.size(), 0.0);
  value_t sum = 0.0;
  std::transform(x.cbegin(), x.cend(), y.begin(), [&](value_t xi) {
    const value_t exi = std::exp(xi);
    sum += exi;
    return exi;
  });
  std::transform(y.cbegin(), y.cend(), y.begin(), [=](value_t yi) {
    return yi/sum;
  }); 
}

void Softmax::deriverate(const array_t& y, array_t& dx) const
{
  // TODO: Add valid implementation of softmax derivative
}

} // namespace transfer
} // namespace ccml