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

void CrossEntropySigmoid::error(const array_t& predicted, const array_t& expected, array_t& error) const
{
  error.resize(predicted.size());
  std::transform(predicted.cbegin(), predicted.cend(), expected.cbegin(), error.begin(), [&](value_t y, value_t y_) {
    return y - y_;
  });
}

void CrossEntropySigmoid::validate(const Network& network) const
{
  AbstractLoss::validate(network);

  // Check that last layer is a softmax layer
  auto transferLayer = std::dynamic_pointer_cast<TransferLayer>(network.outputLayer());
  if((transferLayer == nullptr) || (transferLayer->transfer().name() != "sigmoid"))
  {
    throw std::logic_error("CrossEntropySigmoid is compatible only with networks having Sigmoid output");
  }
  else if(transferLayer->outputSize() != 1) 
  {
    throw std::logic_error("CrossEntropySigmoid is compatible only with networks having single output");
  }
}

bool CrossEntropySigmoid::includesTransfer() const
{
  return true;
}

} // namespace loss
} // namespace ccml