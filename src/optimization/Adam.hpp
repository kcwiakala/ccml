#ifndef CCML_ADAM_HPP
#define CCML_ADAM_HPP

#include "SgdExtension.hpp"

namespace ccml {

struct AdamNeuronData
{
  AdamNeuronData(const Neuron& neuron);

  void reset();

  array_t mW;
  value_t mB;
  array_t vW;
  value_t vB;
};

class Adam: public SgdExtension<AdamNeuronData>
{
public:
  Adam(Network& network, const loss_ptr_t& loss, double rate, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-6);

  virtual ~Adam() {}

protected:
  virtual void adjustNeuron(Neuron& neuron, GradientData& gradients, size_t layerIdx, size_t neuronIdx);

  virtual void initTraining();

  virtual void initEpoch();

private:
  const double _beta1;
  const double _beta2;
  const double _epsilon;

  double _bias1;
  double _bias2;
  size_t _t;
};

} // namespace ccml

#endif // CCML_ADAM_HPP