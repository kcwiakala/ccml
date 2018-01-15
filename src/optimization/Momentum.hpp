#ifndef CCML_MOMENTUM_HPP
#define CCML_MOMENTUM_HPP

#include <optimization/StochasticGradientDescent.hpp>

namespace ccml {

class MomentumNeuronData: public NeuronData
{
public:
  MomentumNeuronData(const Neuron& neuron);

  virtual void reset();

  array_t deltaWeight;
  value_t deltaBias;
};

class Momentum: public StochasticGradientDescent
{
public:
  Momentum(Network& network, loss_ptr_t loss, double rate, double momentum);

  virtual ~Momentum();

protected:
  virtual void adjustNeuron(Neuron& neuron, GradientData& gradients, NeuronData& neuronData);

  virtual neuron_data_ptr_t createNeuronData(const Neuron& neuron) const;

  virtual MomentumNeuronData& neuronData(size_t layerIdx, size_t neuronIdx);

private:

  const double _momentum;
};

} // namespace ccml

#endif // CCML_MOMENTUM_HPP