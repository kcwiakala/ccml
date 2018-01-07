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

double Neuron::net(const std::vector<double>& input) const
{
  assert(input.size() == _weights.size());
  return std::inner_product(input.begin(), input.end(), _weights.begin(), _bias);
}

double Neuron::output(const std::vector<double>& input) const
{
  return _activation(net(input));
}

void Neuron::adjust(const std::vector<double>& deltaWeight, double deltaBias)
{
  assert(deltaWeight.size() == _weights.size());
  std::transform(deltaWeight.begin(), deltaWeight.end(), _weights.begin(), _weights.begin(), std::plus<double>());
  _bias += deltaBias;
}

} // namespace ccml
