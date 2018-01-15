#include <iostream>
#include <algorithm>
#include <Utils.hpp>
#include <optimization/Momentum.hpp>

namespace ccml {

using namespace std::placeholders;

MomentumNeuronData::MomentumNeuronData(const Neuron& neuron):
  deltaWeight(neuron.weights().size(), 0.0),
  deltaBias(0.0)
{
}

void MomentumNeuronData::reset()
{
  deltaBias = 0.0;
  std::fill(deltaWeight.begin(), deltaWeight.end(), 0.0);
}

Momentum::Momentum(Network& network, loss_ptr_t loss, double rate, double momentum):
  StochasticGradientDescent(network, loss, rate),
  _momentum(momentum)
{
    
}

Momentum::~Momentum()
{
}

void Momentum::adjustNeuron(Neuron& neuron, GradientData& gradients, NeuronData& data)
{
  MomentumNeuronData& momentumData = static_cast<MomentumNeuronData&>(data);
  
  const array_t& wg = gradients.weights;
  array_t& dw = momentumData.deltaWeight;
  std::transform(wg.begin(), wg.end(), dw.begin(), dw.begin(), [&](value_t wgi, value_t dwi) {
    return wgi * _rate + dwi * _momentum;
  });
  momentumData.deltaBias = gradients.bias * _rate + momentumData.deltaBias * _momentum;

  neuron.adjust(momentumData.deltaWeight, momentumData.deltaBias);
}

neuron_data_ptr_t Momentum::createNeuronData(const Neuron& neuron) const
{
  return std::make_unique<MomentumNeuronData>(neuron);
}

MomentumNeuronData& Momentum::neuronData(size_t layerIdx, size_t neuronIdx)
{
  return static_cast<MomentumNeuronData&>(StochasticGradientDescent::neuronData(layerIdx, neuronIdx));
}

} // namespace ccml