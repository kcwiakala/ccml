#include <algorithm>
#include <cassert>
#include <numeric>

#include "Neuron.hpp"

namespace ccml {

Neuron::Neuron(size_t inputSize, const Activation& activation):
  _activation(activation),
  _bias(0.0), 
  _weights(inputSize, 0.0)
{
}

size_t Neuron::size() const
{
  return _weights.size();
}

void Neuron::init(const Initializer& weightInit, const Initializer& biasInit)
{
  std::generate(_weights.begin(), _weights.end(), weightInit);
  _bias = biasInit();
}

void Neuron::init(const Initializer& initializer)
{
  init(initializer, initializer);
}

double Neuron::net(const array_t& input) const
{
  assert(input.size() == _weights.size());
  return std::inner_product(input.begin(), input.end(), _weights.begin(), _bias);
}

double Neuron::output(const array_t& input) const
{
  return _activation(net(input));
}

void Neuron::adjust(const array_t& deltaWeight, double deltaBias)
{
  assert(deltaWeight.size() == _weights.size());
  std::transform(deltaWeight.begin(), deltaWeight.end(), _weights.begin(), _weights.begin(), std::plus<double>());
  _bias += deltaBias;
}

} // namespace ccml
