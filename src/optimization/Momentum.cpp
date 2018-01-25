#include <iostream>
#include <algorithm>
#include <Utils.hpp>
#include <optimization/Momentum.hpp>

namespace ccml {

using namespace std::placeholders;

MomentumNodeData::MomentumNodeData(const Node& node):
  deltaWeight(node.size(), 0.0),
  deltaBias(0.0)
{
}

void MomentumNodeData::reset()
{
  deltaBias = 0.0;
  std::fill(deltaWeight.begin(), deltaWeight.end(), 0.0);
}

void Momentum::adjustNode(Node& node, GradientData& gradients, size_t layerIdx, size_t nodeIdx)
{
  MomentumNodeData& data = nodeData(layerIdx, nodeIdx);
  
  const array_t& wg = gradients.weights;
  array_t& dw = data.deltaWeight;
  std::transform(wg.cbegin(), wg.cend(), dw.cbegin(), dw.begin(), [&](value_t wgi, value_t dwi) {
    return wgi * _rate + dwi * _momentum;
  });
  data.deltaBias = gradients.bias * _rate + data.deltaBias * _momentum;

  node.adjust(data.deltaWeight, data.deltaBias);
}

} // namespace ccml