#include <algorithm>
#include <functional>
#include <iostream>
#include <random>

#include <layer/NeuronLayer.hpp>
#include <Utils.hpp>
#include "StochasticGradientDescent.hpp"

namespace ccml {

using namespace std::placeholders;

GradientData::GradientData(const Neuron& neuron):
  weights(neuron.weights().size(), 0.0),
  bias(0.0)
{
}

void GradientData::reset()
{
  bias = 0.0;
  std::fill(weights.begin(), weights.end(), 0.0);
}

StochasticGradientDescent::StochasticGradientDescent(Network& network, const loss_ptr_t& loss, double rate):
  Backpropagation(network, loss), _rate(rate)
{
  _gradients.resize(_network.size());

  for(size_t layerIdx=0; layerIdx < _network.size(); ++layerIdx)
  {
    neuron_layer_ptr_t layer = _network.neuronLayer(layerIdx);
    if(layer)
    {
      _gradients[layerIdx].reserve(layer->size());
      for(size_t neuronIdx=0; neuronIdx < layer->size(); ++neuronIdx)
      {
        _gradients[layerIdx].emplace_back(GradientData(layer->neuron(neuronIdx)));
      }
    }
  }
}

StochasticGradientDescent::~StochasticGradientDescent() 
{
}

void StochasticGradientDescent::updateGradients(const array_t& input, const array_2d_t& activation, const array_2d_t& error)
{
  for(size_t layerIdx=0; layerIdx < _network.size(); ++layerIdx)
  {
    neuron_layer_ptr_t layer = _network.neuronLayer(layerIdx);
    if(layer)
    {
      const array_t& layerInput = (layerIdx == 0) ? input : activation[layerIdx - 1];
      layer->splitError(layerInput, error[layerIdx], [&](Neuron& neuron, const array_t& inputError, size_t neuronIdx) {
        GradientData& data = _gradients[layerIdx][neuronIdx];
        std::transform(inputError.begin(), inputError.end(), 
            data.weights.begin(), data.weights.begin(), 
            std::plus<value_t>());
        data.bias += error[layerIdx][neuronIdx];
      });
    }
  }
}

void StochasticGradientDescent::adjustNeurons()
{
  for(size_t layerIdx=0; layerIdx < _network.size(); ++layerIdx)
  {
    neuron_layer_ptr_t layer = _network.neuronLayer(layerIdx);
    if(layer)
    {
      for(size_t neuronIdx=0; neuronIdx < layer->size(); ++neuronIdx)
      {
        GradientData& gradients = _gradients[layerIdx][neuronIdx];
        adjustNeuron(layer->neuron(neuronIdx), gradients, layerIdx, neuronIdx);
        gradients.reset();
      }
    }
  }
}

void StochasticGradientDescent::adjustNeuron(Neuron& neuron, GradientData& gradients, size_t layerIdx, size_t neuronIdx)
{
  array_t& wg = gradients.weights;
  std::transform(wg.begin(), wg.end(), wg.begin(), [this](value_t wgi) {
    return wgi * _rate;
  });
  neuron.adjust(wg, gradients.bias * _rate);
}

void StochasticGradientDescent::learnSample(const Sample& sample)
{
  thread_local static array_2d_t activation;
  thread_local static array_2d_t error;

  // Feed forward and backpropagate error
  passSample(sample, activation, error);

  // Update gradients for each neuron in the network
  updateGradients(sample.input, activation, error);
}

bool StochasticGradientDescent::train(const sample_list_t& samples, size_t maxIterations, double epsilon)
{
  initNeuronData();

  bool success(false);

  auto sampleIdx = std::bind(std::uniform_int_distribution<size_t>(0, samples.size() - 1), std::default_random_engine());

  for(size_t i=0; i<maxIterations; ++i)
  {
    //learnSample(samples[sampleIdx()]);
    for(size_t j=0; j<samples.size(); ++j) 
    {
      learnSample(samples[j]);
    }
    adjustNeurons();

    const double totalLoss = _loss->compute(_network, samples);
    if(totalLoss < epsilon)
    {
      success = true;
      break;
    }
  }

  return success;
}

neuron_data_ptr_t StochasticGradientDescent::createNeuronData(const Neuron& neuron) const
{
  return neuron_data_ptr_t();
}

NeuronData& StochasticGradientDescent::neuronData(size_t layerIdx, size_t neuronIdx)
{
  return *(_neuronData.at(layerIdx).at(neuronIdx));
}

void StochasticGradientDescent::initNeuronData()
{
  if(_neuronData.empty())
  {
    _neuronData.resize(_network.size());
    for(size_t layerIdx=0; layerIdx<_network.size(); ++layerIdx) 
    {
      neuron_layer_ptr_t neuronLayer = _network.neuronLayer(layerIdx);
      if(neuronLayer)
      {
        _neuronData[layerIdx].reserve(neuronLayer->outputSize());
        neuronLayer->forEachNeuron([&](const Neuron& neuron, size_t j) {
          _neuronData[layerIdx].emplace_back(createNeuronData(neuron));
        });
      }
    }
  }
  else
  {
    for_each(_neuronData, std::bind(&NeuronData::reset, _1));
  }
}

} // namespace ccml