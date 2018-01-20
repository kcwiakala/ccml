#include <algorithm>
#include <cmath>

#include <iostream>

#include "Adam.hpp"

namespace ccml {

AdamNeuronData::AdamNeuronData(const Neuron& neuron):
  mW(neuron.size(), 0.0), mB(0.0), vW(neuron.size(), 0.0), vB(0.0)
{
}

void AdamNeuronData::reset()
{
  mB = 0.0;
  std::fill(mW.begin(), mW.end(), 0.0);
  vB = 0.0;
  std::fill(vW.begin(), vW.end(), 0.0);
}

Adam::Adam(Network& network, const loss_ptr_t& loss, double rate, double beta1, double beta2, double epsilon):
  SgdExtension(network, loss, rate), 
  _beta1(beta1), _beta2(beta2), _epsilon(epsilon), 
  _lr(0.0), _t(0)
{ 
}

void Adam::adjustNeuron(Neuron& neuron, GradientData& gradients, size_t layerIdx, size_t neuronIdx)
{
  AdamNeuronData& data = neuronData(layerIdx, neuronIdx);

  for(size_t i=0; i<gradients.weights.size(); ++i)
  {
    data.mW[i] = _beta1 * data.mW[i] + (1 - _beta1) * gradients.weights[i];
    data.vW[i] = _beta2 * data.vW[i] + (1 - _beta2) * std::pow(gradients.weights[i], 2);
    gradients.weights[i] = _lr * data.mW[i] / (std::sqrt(data.vW[i] + _epsilon));
  }

  data.mB = _beta1 * data.mB + (1 - _beta1) * gradients.bias;
  data.vB = _beta2 * data.vB + (1 - _beta2) * std::pow(gradients.bias, 2);
  gradients.bias = _lr * data.mB / (std::sqrt(data.vB + _epsilon));

  neuron.adjust(gradients.weights, gradients.bias);
}

void Adam::initTraining()
{
  SgdExtension::initTraining();
  _t = 0;
}

void Adam::initEpoch()
{
  SgdExtension::initEpoch();
  ++_t;
  const double bias1 = 1 - std::pow(_beta1, _t);
  const double bias2 = 1 - std::pow(_beta2, _t);
  _lr = _rate * std::sqrt(bias2) / bias1;
}

} // namespace ccml