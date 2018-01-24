#include <algorithm>
#include <numeric>

#include <iostream>

#include "FullyConnectedLayer.hpp"

namespace ccml {

using namespace std::placeholders;

FullyConnectedLayer::FullyConnectedLayer(size_t inputSize, size_t outputSize, const transfer_ptr_t& transfer):
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
  static thread_local array_t aux;

  aux.resize(_nodes.size());
  std::transform(_nodes.cbegin(), _nodes.cend(), aux.begin(), [&](const Node& node) {
    return node.output(x);
  });
  // std::cout << *this << std::endl;
  // std::cout << "IN:  " << x << std::endl;
  // std::cout << "NET: " << aux << std::endl;
  _transfer->apply(aux, y);
  // std::cout << "OUT: " << y << std::endl;
}

void FullyConnectedLayer::backpropagate(const array_t& error, array_t& inputError) const
{
  inputError.resize(_inputSize, 0.0);
  for(size_t i=0; i<_nodes.size(); ++i) 
  {
    const value_t& neuronError = error[i];
    const array_t& weights = _nodes[i].weights();
    for(size_t j=0; j<_inputSize; ++j) 
    {
      inputError[j] += neuronError * weights[j];
    }
  } 
}

void FullyConnectedLayer::splitError(const array_t& x, const array_t& error, const error_reader_t& reader) const
{
  thread_local static array_t aux;
  
  aux.resize(x.size());
  for(size_t i=0; i<_nodes.size(); ++i)
  {
    for(size_t j=0; j<_inputSize; ++j)
    {
      aux[j] = x[j] * error[i];
    }
    reader(aux, i);
  }
}

} // namespace ccml