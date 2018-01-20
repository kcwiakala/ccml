#ifndef CCML_SGD_EXTENSION
#define CCML_SGD_EXTENSION

#include <algorithm>
#include <functional>

#include "StochasticGradientDescent.hpp"

namespace ccml {

template<typename NodeData>
class SgdExtension: public StochasticGradientDescent
{
protected:
  SgdExtension(Network& network, const loss_ptr_t& loss, double rate);

  const NodeData& nodeData(size_t layerIdx, size_t nodeIdx) const;

  NodeData& nodeData(size_t layerIdx, size_t nodeIdx);

  virtual void initTraining();

private:
  vector_2d<NodeData> _nodeData;
};

template<typename NodeData>
SgdExtension<NodeData>::SgdExtension(Network& network, const loss_ptr_t& loss, double rate):
  StochasticGradientDescent(network, loss, rate)
{
  _nodeData.resize(_network.size());
  for(size_t layerIdx=0; layerIdx < _network.size(); ++layerIdx)
  {
    neuron_layer_ptr_t layer = _network.neuronLayer(layerIdx);
    if(layer)
    {
      _nodeData[layerIdx].reserve(layer->size());
      for(size_t nodeIdx=0; nodeIdx < layer->size(); ++nodeIdx)
      {
        _nodeData[layerIdx].emplace_back(NodeData(layer->node(nodeIdx)));
      }
    }
  }
}

template<typename NodeData>
const NodeData& SgdExtension<NodeData>::nodeData(size_t layerIdx, size_t nodeIdx) const
{
  return _nodeData[layerIdx][nodeIdx];
}

template<typename NodeData>
NodeData& SgdExtension<NodeData>::nodeData(size_t layerIdx, size_t nodeIdx)
{
  return _nodeData[layerIdx][nodeIdx];
}

template<typename NodeData>
void SgdExtension<NodeData>::initTraining()
{
  std::for_each(_nodeData.begin(), _nodeData.end(), [](auto& layerData) {
    std::for_each(layerData.begin(), layerData.end(), std::bind(&NodeData::reset, std::placeholders::_1));
  });
}

} // namespace ccml

#endif // CCML_SGD_EXTENSION