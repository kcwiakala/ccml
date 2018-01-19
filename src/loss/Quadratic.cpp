#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

#include <Network.hpp>
#include <loss/Quadratic.hpp>

namespace ccml {
namespace loss {

value_t Quadratic::compute(const Network& network, const Sample& sample) const
{
  thread_local static array_t aux;

  network.output(sample.input, aux);

  std::transform(aux.begin(), aux.end(), sample.output.begin(), aux.begin(), [](value_t output, value_t expected) {
    return std::pow(expected - output, 2.0) / 2;
  });
  return std::accumulate(aux.begin(), aux.end(), 0.0);
}

value_t Quadratic::error(value_t predicted, value_t expected) const
{
  return (expected - predicted);
}

} // namespace loss
} // namespace ccml