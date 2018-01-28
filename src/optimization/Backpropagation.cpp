#include <algorithm>
#include <iostream>
#include <functional>

#include "Backpropagation.hpp"

namespace ccml {

using namespace std::placeholders;

Backpropagation::Backpropagation(Network& network, loss_ptr_t loss):
    NetworkOptimization(network, std::move(loss))
{
}

void Backpropagation::passSample(const Sample& sample, array_2d_t& activation, array_2d_t& error) const
{
  error.resize(_network.size());

  // Feed forward the sample
  feed(sample, activation, error.back());
  
  // Backpropagate the error
  backpropagate(activation, error);
}

void Backpropagation::feed(const Sample& sample, array_2d_t& activation, array_t& error) const
{
  _network.output(sample.input, activation);

  const array_t& output = activation.back();

  _loss->error(output, sample.output, error);
  if(!_transferIncluded)
  {
     array_t aux;  
    _network.outputLayer()->error(output, error, aux);
    error.swap(aux);
  }
}

void Backpropagation::backpropagate(const array_2d_t& activation, array_2d_t& error) const
{
  static array_t aux;

  size_t layerIdx = _network.size()-1;
  while(layerIdx-- > 0)
  {
    // Backpropagate error from next level
    _network.layer(layerIdx+1)->backpropagate(error[layerIdx+1], error[layerIdx]);

    // Calculate error term for the layer
    _network.layer(layerIdx)->error(activation[layerIdx], error[layerIdx], aux);
    error[layerIdx].swap(aux);    
  }
}

} // namespace ccml