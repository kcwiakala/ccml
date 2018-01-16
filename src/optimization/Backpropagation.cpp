#include <algorithm>
#include <functional>

#include "Backpropagation.hpp"

namespace ccml {

using namespace std::placeholders;

Backpropagation::Backpropagation(Network& network, const loss_ptr_t& loss):
  NetworkOptimization(network, loss)
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

  std::transform(output.begin(), output.end(), sample.output.begin(), error.begin(), 
      std::bind(&Loss::error, _loss, _1, _2));
}

void Backpropagation::backpropagate(const array_2d_t& activation, array_2d_t& error) const
{
  thread_local static array_t aux;

  size_t layerIdx = _network.size();
  while(layerIdx-- > 0)
  {
    neuron_layer_ptr_t neuronLayer = _network.neuronLayer(layerIdx);
    if(neuronLayer)
    {
        // Calculate error term for the neuron layer
        neuronLayer->error(activation[layerIdx], error[layerIdx], aux);
        error[layerIdx].swap(aux);
    }
    if(layerIdx > 0)
    {
      _network.layer(layerIdx)->backpropagate(error[layerIdx], error[layerIdx-1]);
    }
  }
}

} // namespace ccml