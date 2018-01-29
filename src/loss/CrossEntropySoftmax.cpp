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
  thread_local array_t aux;

  network.output(sample.input, aux);

  std::transform(aux.begin(), aux.end(), sample.output.begin(), aux.begin(), [](value_t output, value_t expected) {
    return -(expected * std::log(output));
  });
  return std::accumulate(aux.begin(), aux.end(), 0.0);
}

void CrossEntropySoftmax::error(const array_t& predicted, const array_t& expected, array_t& error) const
{
  error.resize(predicted.size());
  std::transform(predicted.cbegin(), predicted.cend(), expected.cbegin(), error.begin(), [&](value_t y, value_t y_) {
    return y - y_;
  });
}

void CrossEntropySoftmax::validate(const Network& network) const
{
  AbstractLoss::validate(network);

  // Check that last layer is a softmax layer
  auto transferLayer = std::dynamic_pointer_cast<TransferLayer>(network.outputLayer());
  if((transferLayer == nullptr) || (transferLayer->transfer().name() != "softmax"))
  {
    throw std::logic_error("CrossEntropySoftmax is compatible only with networks having SoftmaxLayer output");
  }
}

bool CrossEntropySoftmax::includesTransfer() const
{
  return true;
}

} // namespace loss
} // namespace ccml