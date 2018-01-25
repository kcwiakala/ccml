#ifndef CCML_MOMENTUM_HPP
#define CCML_MOMENTUM_HPP

#include "SgdExtension.hpp"

namespace ccml {

struct MomentumNodeData
{
  MomentumNodeData(const Node& node);

  void reset();

  array_t deltaWeight;
  value_t deltaBias;
};

class Momentum: public SgdExtension<MomentumNodeData>
{
public:
  template<typename Tl>
  Momentum(Network& network, Tl&& loss, double rate, double momentum):
    SgdExtension(network, std::forward<Tl>(loss), rate),
    _momentum(momentum)
  {}

protected:
  virtual void adjustNode(Node& node, GradientData& gradients, size_t layerIdx, size_t neuronIdx);

private:
  const double _momentum;
};

} // namespace ccml

#endif // CCML_MOMENTUM_HPP