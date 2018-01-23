#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

#include <Network.hpp>
#include <layer/TransferLayer.hpp>
#include <loss/CrossEntropySigmoid.hpp>

namespace ccml {
namespace loss {

value_t CrossEntropySigmoid::compute(const Network& network, const Sample& sample) const
{
  thread_local static array_t aux;

  network.output(sample.input, aux);

  std::transform(aux.begin(), aux.end(), sample.output.cbegin(), aux.begin(), [](value_t predicted, value_t expected) {
    return -(expected * std::log(predicted) + (1 - expected) * std::log(1 - predicted));
  });
  return std::accumulate(aux.begin(), aux.end(), 0.0) / aux.size();

}

value_t CrossEntropySigmoid::error(value_t predicted, value_t expected) const
{
  const value_t denominator = predicted * (1 - predicted);
  const value_t difference = predicted - expected;
  return (denominator < 1e-9) ? difference : (difference / denominator);
}

void CrossEntropySigmoid::validate(const Network& network) const
{
  Loss::validate(network);

  // Check that last layer is a softmax layer
  layer_ptr_t layer = network.layer(network.size() - 1);
  auto transferLayer = dynamic_cast<const TransferLayer*>(layer.get());
  if((transferLayer == nullptr) || !(transferLayer->transfer() == transfer::sigmoid()))
  {
    throw std::logic_error("CrossEntropySigmoid is compatible only with networks having SigmoidLayer output");
  }
}

} // namespace loss
} // namespace ccml