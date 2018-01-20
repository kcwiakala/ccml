#include <algorithm>
#include <cassert>
#include <numeric>

#include <iostream>

#include "Neuron.hpp"

namespace ccml {

Neuron::Neuron(size_t inputSize, const Transfer& transfer):
  Node(inputSize),
  _transfer(transfer)
{
}

value_t Neuron::output(const array_t& input) const
{
  return _transfer.operation(Node::output(input));
}

void Neuron::toStream(std::ostream& stream) const
{
  stream << "{w:[";
  for(size_t i=0; i<_weights.size(); ++i) 
  {
    stream << ((i>0) ? "," : "") << _weights[i];
  }
  stream << "],b:" << _bias << ",a:" << _transfer.name << "}";
}

} // namespace ccml