#include <algorithm>

#include "NeuronLayer.hpp"

#include <Utils.hpp>

namespace ccml {

using namespace std::placeholders;

NeuronLayer::NeuronLayer(const std::string& type, size_t layerSize, size_t neuronSize, const Activation& activation):
  TypedLayer(type),
  _neurons(layerSize, Neuron(neuronSize, activation))
{
}

size_t NeuronLayer::outputSize() const
{
  return _neurons.size();
}

void NeuronLayer::forEachNeuron(const neuron_updater_t& updater)
{
  indexed_for_each(_neurons.begin(), _neurons.end(), updater);
}

void NeuronLayer::forEachNeuron(const neuron_reader_t& reader) const
{
  indexed_for_each(_neurons.begin(), _neurons.end(), reader);
}

void NeuronLayer::init(const Initializer& weightInit, const Initializer& biasInit)
{
  std::for_each(_neurons.begin(), _neurons.end(), std::bind(&Neuron::init, _1, weightInit, biasInit));
}

void NeuronLayer::toStream(std::ostream& stream) const
{
  stream << type() << ":{is:" << inputSize() << ",os:" << outputSize();
  stream << ",n:[";
  for(size_t i=0; i<_neurons.size(); ++i) 
  {
    stream << ((i>0) ? "," : "") << _neurons[i];
  }
  stream << "]}";
}

} // namespace ccml