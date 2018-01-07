#include <algorithm>
#include <functional>

#include <Layer.hpp>

namespace ccml {

void Layer::init(const Initializer& weightInit, const Initializer& biasInit)
{
  std::for_each(_neurons.begin(), _neurons.end(), [&](Neuron& n) {
    n.init(weightInit, biasInit);
  });
}

void Layer::init(const Initializer& initializer)
{
  init(initializer, initializer);
}

std::vector<double> Layer::output(const std::vector<double>& x)
{
  std::vector<double> result;
  result.reserve(_neurons.size());
  std::transform(_neurons.begin(), _neurons.end(), std::back_inserter(result), [&](Neuron& n) {
    return n.output(x);
  });
  return result;
}

} // namespace ccml