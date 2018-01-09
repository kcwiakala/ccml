#include "NeuronLayer.hpp"

namespace ccml {

NeuronLayer::NeuronLayer(size_t layerSize, size_t neuronSize, const Activation& activation):
  _neurons(layerSize, Neuron(neuronSize, activation))
{
}

size_t NeuronLayer::outputSize() const
{
  return _neurons.size();
}

} // namespace ccml