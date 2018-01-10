#include <algorithm>

#include "NeuronLayer.hpp"

namespace ccml {
namespace {

template<typename Cont, typename F>
void applyForEach(Cont& c, const F& fun)
{
  for(size_t i=0; i<c.size(); ++i) 
  {
    fun(c[i], i);
  }
}

} // namespace

using namespace std::placeholders;

NeuronLayer::NeuronLayer(size_t layerSize, size_t neuronSize, const Activation& activation):
  _neurons(layerSize, Neuron(neuronSize, activation))
{
}

size_t NeuronLayer::outputSize() const
{
  return _neurons.size();
}

void NeuronLayer::forEachNeuron(const neuron_updater_t& updater)
{
  applyForEach(_neurons, updater);
}

void NeuronLayer::forEachNeuron(const neuron_reader_t& reader) const
{
  applyForEach(_neurons, reader);
}

void NeuronLayer::init(const Initializer& weightInit, const Initializer& biasInit)
{
  std::for_each(_neurons.begin(), _neurons.end(), std::bind(&Neuron::init, _1, weightInit, biasInit));
}

} // namespace ccml