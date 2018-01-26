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

  std::transform(aux.cbegin(), aux.cend(), sample.output.cbegin(), aux.begin(), [](value_t output, value_t expected) {
    return std::pow(expected - output, 2.0) / 2;
  });
  return std::accumulate(aux.cbegin(), aux.cend(), 0.0) / aux.size();
}

void MeanSquaredError::error(const array_t& predicted, const array_t& expected, array_t& error) const
{
  error.resize(predicted.size());
  std::transform(predicted.cbegin(), predicted.cend(), expected.cbegin(), error.begin(), [&](value_t y, value_t y_) {
    return y - y_;
  });
}

} // namespace loss
} // namespace ccml