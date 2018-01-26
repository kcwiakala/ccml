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
  error.resize(output.size());

  // Calclate error on output layer
  std::transform(output.cbegin(), output.cend(), sample.output.cbegin(), error.begin(), [&](value_t predicted, value_t expected) {
    return _loss->error(predicted, expected);
  });
}

void Backpropagation::backpropagate(const array_2d_t& activation, array_2d_t& error) const
{
  thread_local static array_t aux;

  size_t layerIdx = _network.size();
  while(layerIdx-- > 0)
  {
    layer_ptr_t layer = _network.layer(layerIdx);

    // Calculate error term for the neuron layer
    layer->error(activation[layerIdx], error[layerIdx], aux);
    // std::cout << "ERROR: " << aux << std::endl;
    error[layerIdx].swap(aux);

    if(layerIdx > 0)
    {
      layer->backpropagate(error[layerIdx], error[layerIdx-1]);
    }
  }
}

} // namespace ccml