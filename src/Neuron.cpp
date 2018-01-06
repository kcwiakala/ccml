#include <algorithm>
#include <cassert>
#include <numeric>

#include "Neuron.hpp"

namespace ccml {

Neuron::Neuron(size_t inputSize, Activation& activation):
  _activation(activation),
  _net(0.0), 
  _value(0.0), 
  _bias(0.0), 
  _weights(inputSize, 0.0)
{
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

double Neuron::activate(const std::vector<double>& input)
{
  _net = net(input);
  _value = _activation(_net);
  return _value;
}

double Neuron::errorGradient(double error) const
{

}

} // namespace ccml
