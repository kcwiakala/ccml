#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

#include <Network.hpp>
#include <loss/MeanSquaredError.hpp>

namespace ccml {
namespace loss {

value_t MeanSquaredError::compute(const Network& network, const Sample& sample) const
{
  thread_local static array_t aux;

  network.output(sample.input, aux);

  std::transform(aux.begin(), aux.end(), sample.output.begin(), aux.begin(), [](value_t output, value_t expected) {
    return std::pow(expected - output, 2.0) / 2;
  });
  return std::accumulate(aux.begin(), aux.end(), 0.0) / aux.size();
}

value_t MeanSquaredError::error(value_t predicted, value_t expected) const
{
  return (predicted - expected);
}

} // namespace loss
} // namespace ccml