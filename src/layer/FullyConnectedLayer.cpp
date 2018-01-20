#include <algorithm>
#include <numeric>

#include "FullyConnectedLayer.hpp"

namespace ccml {

using namespace std::placeholders;

FullyConnectedLayer::FullyConnectedLayer(size_t inputSize, size_t outputSize, const Transfer& transfer):
  NeuronLayer("FullyConnectedLayer", outputSize, inputSize, transfer),
  _inputSize(inputSize)
{
}

size_t FullyConnectedLayer::inputSize() const
{
  return _inputSize;
}

void FullyConnectedLayer::output(const array_t& x, array_t& y) const
{
  y.resize(_neurons.size());
  std::transform(_neurons.begin(), _neurons.end(), y.begin(), [&](const Neuron& neuron) {
    return _transfer.operation(neuron.Node::output(x));
  });
}

void FullyConnectedLayer::backpropagate(const array_t& error, array_t& inputError) const
{
  inputError.resize(_inputSize, 0.0);
  for(size_t i=0; i<_neurons.size(); ++i) 
  {
    const value_t& neuronError = error[i];
    const array_t& weights = _neurons[i].weights();
    for(size_t j=0; j<_inputSize; ++j) 
    {
      inputError[j] += neuronError * weights[j];
    }
  } 
}

void FullyConnectedLayer::splitError(const array_t& x, const array_t& error, const neuron_adjuster_t& adjuster)
{
  thread_local static array_t aux;
  
  aux.resize(x.size());
  for(size_t i=0; i<_neurons.size(); ++i)
  {
    for(size_t j=0; j<_inputSize; ++j)
    {
      aux[j] = x[j] * error[i];
    }
    adjuster(_neurons[i], aux, i);
  }
}

} // namespace ccml