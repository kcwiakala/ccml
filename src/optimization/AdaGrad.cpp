#include <algorithm>
#include <cmath>

#include <optimization/AdaGrad.hpp>

namespace ccml {

using namespace std::placeholders;

AdaGradNodeData::AdaGradNodeData(const Node& node):
  diagWeight(node.size(), 0.0),
  diagBias(0.0)
{
}

void AdaGradNodeData::reset()
{
  diagBias = 0.0;
  std::fill(diagWeight.begin(), diagWeight.end(), 0.0);
}

AdaGrad::AdaGrad(Network& network, loss_ptr_t loss, double rate, double epsilon):
    SgdExtension(network, std::move(loss), rate),
    _epsilon(epsilon)
{
}

void AdaGrad::adjustNode(Node& node, GradientData& gradients, size_t layerIdx, size_t nodeIdx)
{
  AdaGradNodeData& data = nodeData(layerIdx, nodeIdx);
  
  for(size_t i=0; i<gradients.weights.size(); ++i)
  {
    data.diagWeight[i] += std::pow(gradients.weights[i], 2);
    gradients.weights[i] *= _rate / data.diagWeight[i]; 
  }
  data.diagBias += std::pow(gradients.bias, 2);
  gradients.bias *= _rate / data.diagBias;

  node.adjust(gradients.weights, gradients.bias);
}

} // namespace ccml