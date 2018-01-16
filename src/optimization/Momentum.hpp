#ifndef CCML_MOMENTUM_HPP
#define CCML_MOMENTUM_HPP

#include "SgdExtension.hpp"

namespace ccml {

struct MomentumNeuronData
{
  MomentumNeuronData(const Neuron& neuron);

  void reset();

  array_t deltaWeight;
  value_t deltaBias;
};

class Momentum: public SgdExtension<MomentumNeuronData>
{
public:
  Momentum(Network& network, const loss_ptr_t& loss, double rate, double momentum);

  virtual ~Momentum() {}

protected:
  virtual void adjustNeuron(Neuron& neuron, GradientData& gradients, size_t layerIdx, size_t neuronIdx);

private:
  const double _momentum;
};

} // namespace ccml

#endif // CCML_MOMENTUM_HPP