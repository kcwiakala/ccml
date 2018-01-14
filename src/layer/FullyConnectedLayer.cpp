#include <algorithm>
#include <numeric>

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

void FullyConnectedLayer::backpropagate(const array_t& error, array_t& inputError) const
{
  inputError.resize(_inputSize, 0.0);
  for(size_t i=0; i<_neurons.size(); ++i) 
  {
    const array_t& weights = _neurons[i].weights();
    for(size_t j=0; j<_inputSize; ++j) 
    {
      inputError[j] += error[i] * weights[j];
    }
  } 
}

void FullyConnectedLayer::adjust(const array_t& x, const array_t& dx, const neuron_adjuster_t& adjuster)
{
  array_t aux(x.size());
  // for(size_t i=0; i<_neurons.size(); ++i) 
  // {
  //   const array_t& weights = _neurons[i].weights();
  //   std::transform(weights.begin(), weights.end(), aux.begin(), [&](value_t w) {
  //     return w * dx[i];
  //   });
  //   adjuster(_neurons[i], aux, i);
  // }

  for(size_t i=0; i<_neurons.size(); ++i)
  {
    //const array_t& weights = _neurons[i].weights();
    for(size_t j=0; j<_inputSize; ++j)
    {
      aux[j] = x[j] * dx[i];
    }
    adjuster(_neurons[i], aux, i);
  }
}

} // namespace ccml