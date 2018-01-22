#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

#include <Network.hpp>
#include <loss/CrossEntropySoftmax.hpp>

namespace ccml {
namespace loss {

value_t CrossEntropySoftmax::compute(const Network& network, const Sample& sample) const
{
  thread_local static array_t aux;

  network.output(sample.input, aux);

  std::transform(aux.begin(), aux.end(), sample.output.begin(), aux.begin(), [](value_t output, value_t expected) {
    return -(expected * std::log(output));
  });
  return std::accumulate(aux.begin(), aux.end(), 0.0);
}

value_t CrossEntropySoftmax::error(value_t predicted, value_t expected) const
{
  return - expected / std::max(predicted, 1e-9);
}

} // namespace loss
} // namespace ccml