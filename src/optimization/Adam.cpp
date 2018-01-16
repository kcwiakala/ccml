#include <algorithm>
#include <cmath>

#include <iostream>

#include "Adam.hpp"

namespace ccml {

AdamNeuronData::AdamNeuronData(const Neuron& neuron):
  mW(neuron.weights().size(), 0.0), mB(0.0), vW(neuron.weights().size(), 0.0), vB(0.0)
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
  _bias1(0.0), _bias2(0.0), _t(0)
{ 
}

void Adam::adjustNeuron(Neuron& neuron, GradientData& gradients, size_t layerIdx, size_t neuronIdx)
{
  AdamNeuronData& data = neuronData(layerIdx, neuronIdx);

  //std::cout << "ADAM: " << _t << " " << _bias1 << " " << _bias2 << std::endl;

  for(size_t i=0; i<gradients.weights.size(); ++i)
  {
    data.mW[i] = _beta1 * data.mW[i] + (1 - _beta1) * gradients.weights[i];
    data.vW[i] = _beta2 * data.vW[i] + (1 - _beta2) * std::pow(gradients.weights[i], 2);
    gradients.weights[i] = (data.mW[i] / _bias1) * _rate / (std::sqrt(data.vW[i] / _bias2) + _epsilon);
  }

  data.mB = _beta1 * data.mB + (1 - _beta1) * gradients.bias;
  data.vB = _beta2 * data.vB + (1 - _beta2) * std::pow(gradients.bias, 2);
  gradients.bias = (data.mB / _bias1) * _rate / (std::sqrt(data.vB / _bias2) + _epsilon);

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
  _bias1 = 1 - std::pow(_beta1, _t);
  _bias2 = 1 - std::pow(_beta2, _t);
}

} // namespace ccml