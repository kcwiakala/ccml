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

void Layer::output(const array_t& x, array_t& y)
{
  y.clear();
  std::transform(_neurons.begin(), _neurons.end(), std::back_inserter(y), [&](Neuron& n) {
    return n.output(x);
  });
}

} // namespace ccml