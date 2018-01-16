#ifndef CCML_SGD_EXTENSION
#define CCML_SGD_EXTENSION

#include <algorithm>
#include <functional>

#include "StochasticGradientDescent.hpp"

namespace ccml {

template<typename NeuronData>
class SgdExtension: public StochasticGradientDescent
{
protected:
  SgdExtension(Network& network, const loss_ptr_t& loss, double rate);

  const NeuronData& neuronData(size_t layerIdx, size_t neuronIdx) const;

  NeuronData& neuronData(size_t layerIdx, size_t neuronIdx);

  virtual void initTraining();

private:
  vector_2d<NeuronData> _neuronData;
};

template<typename NeuronData>
SgdExtension<NeuronData>::SgdExtension(Network& network, const loss_ptr_t& loss, double rate):
  StochasticGradientDescent(network, loss, rate)
{
  _neuronData.resize(_network.size());
  for(size_t layerIdx=0; layerIdx < _network.size(); ++layerIdx)
  {
    neuron_layer_ptr_t layer = _network.neuronLayer(layerIdx);
    if(layer)
    {
      _neuronData[layerIdx].reserve(layer->size());
      for(size_t neuronIdx=0; neuronIdx < layer->size(); ++neuronIdx)
      {
        _neuronData[layerIdx].emplace_back(NeuronData(layer->neuron(neuronIdx)));
      }
    }
  }
}

template<typename NeuronData>
const NeuronData& SgdExtension<NeuronData>::neuronData(size_t layerIdx, size_t neuronIdx) const
{
  return _neuronData[layerIdx][neuronIdx];
}

template<typename NeuronData>
NeuronData& SgdExtension<NeuronData>::neuronData(size_t layerIdx, size_t neuronIdx)
{
  return _neuronData[layerIdx][neuronIdx];
}

template<typename NeuronData>
void SgdExtension<NeuronData>::initTraining()
{
  std::for_each(_neuronData.begin(), _neuronData.end(), [](auto& layerData) {
    std::for_each(layerData.begin(), layerData.end(), std::bind(&NeuronData::reset, std::placeholders::_1));
  });
}

} // namespace ccml

#endif // CCML_SGD_EXTENSION