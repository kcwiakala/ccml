#ifndef CCML_ADAM_HPP
#define CCML_ADAM_HPP

#include "SgdExtension.hpp"

namespace ccml {

struct AdamNodeData
{
  AdamNodeData(const Node& node);

  void reset();

  array_t mW;
  value_t mB;
  array_t vW;
  value_t vB;
};

class Adam: public SgdExtension<AdamNodeData>
{
public:
  template<typename Tl>
  Adam(Network& network, Tl&& loss, double rate, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8):
    SgdExtension(network, std::forward<Tl>(loss), rate), 
    _beta1(beta1), _beta2(beta2), _epsilon(epsilon), 
    _lr(0.0), _t(0)
  {}

protected:
  virtual void adjustNode(Node& node, GradientData& gradients, size_t layerIdx, size_t neuronIdx);

  virtual void initTraining();

  virtual void initEpoch();

private:
  const double _beta1;
  const double _beta2;
  const double _epsilon;

  double _lr;
  size_t _t;
};

} // namespace ccml

#endif // CCML_ADAM_HPP