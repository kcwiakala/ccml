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

Momentum::Momentum(Network& network, const loss_ptr_t& loss, double rate, double momentum):
  SgdExtension(network, loss, rate),
  _momentum(momentum)
{    
}

void Momentum::adjustNeuron(Neuron& neuron, GradientData& gradients, size_t layerIdx, size_t neuronIdx)
{
  MomentumNeuronData& data = neuronData(layerIdx, neuronIdx);
  
  const array_t& wg = gradients.weights;
  array_t& dw = data.deltaWeight;
  std::transform(wg.begin(), wg.end(), dw.begin(), dw.begin(), [&](value_t wgi, value_t dwi) {
    return wgi * _rate + dwi * _momentum;
  });
  data.deltaBias = gradients.bias * _rate + data.deltaBias * _momentum;

  neuron.adjust(data.deltaWeight, data.deltaBias);
}

} // namespace ccml