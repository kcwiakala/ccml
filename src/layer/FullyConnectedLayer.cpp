#include <algorithm>

#include "FullyConnectedLayer.hpp"

namespace ccml {

using namespace std::placeholders;

FullyConnectedLayer::FullyConnectedLayer(size_t inputSize, size_t outputSize, const Activation& activation):
  NeuronLayer("FullyConnectedLayer", outputSize, inputSize, activation),
  _inputSize(inputSize)
{
}

size_t FullyConnectedLayer::inputSize() const
{
  return _inputSize;
}

void FullyConnectedLayer::output(const array_t& x, array_t& y) 
{
  y.resize(_neurons.size());
  std::transform(_neurons.begin(), _neurons.end(), y.begin(), std::bind(&Neuron::output, _1, x));
}

} // namespace ccml