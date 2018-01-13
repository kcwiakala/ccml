#include <algorithm>

#include <layer/NeuronLayer.hpp>
#include "StochasticGradientDescent.hpp"

namespace ccml {

void StochasticGradientDescent::updateLayer(Network& network, size_t layerIdx, const array_2d_t& activation, array_t& error)
{
  array_t aux;
  layer_ptr_t layer = network.layer(layerIdx);

  // If we are in hidden layer, backpropagate error through next layer
  if(layerIdx < network.size() - 1) 
  {
    network.layer(layerIdx + 1)->backpropagate(error, aux);
    error.swap(aux);
  }

  neuron_layer_ptr_t neuronLayer = std::dynamic_pointer_cast<NeuronLayer>(layer);
  if(neuronLayer)
  {
    // Calculate error gradient
    array_t errorGradient;
    neuronLayer->gradient(activation[layerIdx], error, errorGradient);
    
    // Adjust neurons in the layer
    const array_t& layerInput = activation[layerIdx - 1];
    neuronLayer->adjust(layerInput, errorGradient, [&](Neuron& n, const array_t& weightGradient, size_t i) {
      n.adjust(weightGradient, errorGradient[i]);
    });
  }
}

void StochasticGradientDescent::learnSample(Network& network, const Sample& sample)
{
  if((network.inputSize() != sample.input.size()) || (network.outputSize() != sample.output.size()))
  {
    // ERROR
  }
  array_2d_t activation;
  network.output(sample.input, activation);
  
  const array_t& y = activation.back();
  array_t error(y.size());
  std::transform(y.begin(), y.end(), sample.output.begin(), error.begin(), [](value_t output, value_t expected) {
    return expected - output;
  });

  size_t layerIdx = network.size() - 1;
  while(layerIdx > 0)
  {
    updateLayer(network, layerIdx--, activation, error);
  }
}

} // namespace ccml