#ifndef CCML_ADA_GRAD_HPP
#define CCML_ADA_GRAD_HPP

#include "SgdExtension.hpp"

namespace ccml {

struct AdaGradNodeData
{
  AdaGradNodeData(const Node& node);

  void reset();

  array_t diagWeight;
  value_t diagBias;
};

class AdaGrad: public SgdExtension<AdaGradNodeData>
{
public:
  AdaGrad(Network& network, loss_ptr_t loss, double rate, double epsilon = 1e-6);

protected:
  virtual void adjustNode(Node& node, GradientData& gradients, size_t layerIdx, size_t neuronIdx);

private:
  const double _epsilon;
};

} // namespace ccml

#endif // CCML_ADA_GRAD_HPP