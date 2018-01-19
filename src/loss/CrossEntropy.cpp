#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

#include <Network.hpp>
#include <loss/CrossEntropy.hpp>

namespace ccml {
namespace loss {

value_t CrossEntropy::compute(const Network& network, const Sample& sample) const
{
  thread_local static array_t aux;

  network.output(sample.input, aux);

  std::transform(aux.begin(), aux.end(), sample.output.begin(), aux.begin(), [](value_t output, value_t expected) {
    return -(expected * std::log(output) + (1 - expected) * std::log(1 - output));
  });
  return std::accumulate(aux.begin(), aux.end(), 0.0) / aux.size();
}

value_t CrossEntropy::error(value_t predicted, value_t expected) const
{
  const value_t denominator = predicted * (1 - predicted);
  const value_t difference = predicted - expected;
  return (denominator < 1e-9) ? difference : (difference / denominator);
}

} // namespace loss
} // namespace ccml