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

void Neuron::init(const Initializer& weightInit, const Initializer& biasInit)
{
  std::generate(_weights.begin(), _weights.end(), weightInit);
  _bias = biasInit();
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

void Neuron::dump(std::ostream& stream) const
{
  stream << "{w:[";
  for(size_t i=0; i<_weights.size(); ++i) 
  {
    stream << ((i>0) ? "," : "") << _weights[i];
  }
  stream << "],b:" << _bias << ",a:" << _activation.name() << "}";
}

} // namespace ccml