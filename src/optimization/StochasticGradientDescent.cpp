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

StochasticGradientDescent::StochasticGradientDescent(Network& network, loss_ptr_t loss, double rate):
  _network(network), _loss(loss), _rate(rate)
{
  _gradients.resize(_network.size());
  for(size_t layerIdx=0; layerIdx<_network.size(); ++layerIdx) 
  {
    neuron_layer_ptr_t neuronLayer = _network.neuronLayer(layerIdx);
    if(neuronLayer)
    {
      _gradients[layerIdx].reserve(neuronLayer->outputSize());
      neuronLayer->forEachNeuron([&](const Neuron& neuron, size_t j) {
        _gradients[layerIdx].emplace_back(GradientData(neuron));
      });
    }
  }
}

StochasticGradientDescent::~StochasticGradientDescent() 
{
}

void StochasticGradientDescent::backpropagate(const array_2d_t& activation, array_2d_t& error) const
{
  thread_local static array_t aux;

  size_t layerIdx = _network.size();
  while(layerIdx-- > 0)
  {
    neuron_layer_ptr_t neuronLayer = _network.neuronLayer(layerIdx);
    if(neuronLayer)
    {
        // Calculate error term for the neuron layer
        neuronLayer->error(activation[layerIdx], error[layerIdx], aux);
        error[layerIdx].swap(aux);
    }
    if(layerIdx > 0)
    {
      _network.layer(layerIdx)->backpropagate(error[layerIdx], error[layerIdx-1]);
    }
  }
}

void StochasticGradientDescent::outputError(const array_t& output, const array_t& expected, array_t& error) const
{
  error.resize(output.size());
  std::transform(output.begin(), output.end(), expected.begin(), error.begin(), std::bind(&Loss::error, _loss, _1, _2));
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
      layer->forEachNeuron([&](Neuron& neuron, size_t neuronIdx) {
        GradientData& gradients = _gradients[layerIdx][neuronIdx];
        adjustNeuron(neuron, gradients, neuronData(layerIdx, neuronIdx));
        gradients.reset();
      });
    }
  }
}

void StochasticGradientDescent::adjustNeuron(Neuron& neuron, GradientData& gradients, NeuronData& data)
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

  error.resize(_network.size());

  // Feed forward the sample
  _network.output(sample.input, activation);

  // Calculate output error
  outputError(activation.back(), sample.output, error.back());

  // Backpropagate the error
  backpropagate(activation, error);

  // Update gradients for each neuron in the network
  updateGradients(sample.input, activation, error);

  // Adjust neurons with stored gradients
  adjustNeurons();
}

bool StochasticGradientDescent::train(const sample_list_t& samples, size_t maxIterations, double epsilon)
{
  initNeuronData();

  bool success(false);

  // auto sampleIdx = std::bind(std::uniform_int_distribution<size_t>(0, samples.size() - 1), std::default_random_engine());

  for(size_t i=0; i<maxIterations; ++i)
  {
    //learnSample(samples[sampleIdx()]);
    for(size_t j=0; j<samples.size(); ++j) 
    {
      learnSample(samples[j]);
    }

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