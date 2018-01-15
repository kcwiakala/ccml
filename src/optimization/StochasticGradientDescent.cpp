#include <algorithm>
#include <functional>
#include <iostream>
#include <random>

#include <layer/NeuronLayer.hpp>
#include <Utils.hpp>
#include "StochasticGradientDescent.hpp"

namespace ccml {

using namespace std::placeholders;


SgdNeuronData::SgdNeuronData(const Neuron& neuron):
  weightGradient(neuron.weights().size(), 0.0),
  biasGradient(0.0)
{
}

void SgdNeuronData::reset()
{
  biasGradient = 0.0;
  std::fill(weightGradient.begin(), weightGradient.end(), 0.0);
}

StochasticGradientDescent::StochasticGradientDescent(Network& network, loss_ptr_t loss, double rate):
  _network(network), _loss(loss), _rate(rate)
{
}

StochasticGradientDescent::~StochasticGradientDescent()
{
}

void StochasticGradientDescent::adjust(neuron_layer_ptr_t layer, const array_t& input, 
  const array_t& error, size_t layerIndex)
{
  thread_local static array_t aux; 

  layer->splitError(input, error, [&](Neuron& n, const array_t& wg, size_t i) {
    aux.resize(wg.size());
    std::transform(wg.begin(), wg.end(), aux.begin(), [=](value_t wgi) {
      return wgi * _rate;
    });
    n.adjust(aux, error[i] * _rate);
  });
}

void StochasticGradientDescent::updateLayer(size_t layerIdx, const array_t& input, const array_t& output, array_t& error)
{
  thread_local static array_t aux;
  
  neuron_layer_ptr_t neuronLayer = _network.neuronLayer(layerIdx);
  if(neuronLayer)
  {
    // Calculate error term for layer
    neuronLayer->error(output, error, aux);
    error.swap(aux);
    
    // If it's not the first layer calculate backpropagated error
    if(layerIdx > 0) 
    {
      neuronLayer->backpropagate(error, aux);  
    }
    
    // Adjust neurons in the layer
    adjust(neuronLayer, input, error, layerIdx);

    // Return backpropagated error for updating next layers
    error.swap(aux);
  }
  else if(layerIdx > 0)
  {
    layer_ptr_t layer = _network.layer(layerIdx);
    // Calculate backpropagated error and return it for next layers
    layer->backpropagate(error, aux);
    error.swap(aux);
  }
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

void StochasticGradientDescent::adjust(const array_t& input, const array_2d_t& activation, const array_2d_t& error)
{
  for(size_t layerIdx=0; layerIdx < _network.size(); ++layerIdx)
  {
    neuron_layer_ptr_t layer = _network.neuronLayer(layerIdx);
    if(layer)
    {
      const array_t& layerInput = (layerIdx == 0) ? input : activation[layerIdx - 1];
      // ...
    }
  }
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
        SgdNeuronData& data = neuronData(layerIdx, neuronIdx);
        std::transform(inputError.begin(), inputError.end(), 
            data.weightGradient.begin(), data.weightGradient.begin(), std::plus<value_t>());
        data.biasGradient += error[layerIdx][neuronIdx];
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
        SgdNeuronData& data = neuronData(layerIdx, neuronIdx);
        adjustNeuron(neuron, data);
        data.SgdNeuronData::reset();
      });
    }
  }
}

void StochasticGradientDescent::adjustNeuron(Neuron& neuron, NeuronData& data)
{
  SgdNeuronData& sgdData = static_cast<SgdNeuronData&>(data);
  array_t& wg = sgdData.weightGradient;
  std::transform(wg.begin(), wg.end(), wg.begin(), [this](value_t wgi) {
    return wgi * _rate;
  });
  neuron.adjust(wg, sgdData.biasGradient * _rate);
}

void StochasticGradientDescent::learnSample2(const Sample& sample)
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

  //adjust(sample.input, activation, error);

  // Update gradients for each neuron in the network
  updateGradients(sample.input, activation, error);

  // Adjust neurons with stored gradients
  adjustNeurons();
}

void StochasticGradientDescent::learnSample(const Sample& sample)
{
  //assert((network.inputSize() != sample.input.size()) || (network.outputSize() != sample.output.size()))
  array_2d_t activation;
  _network.output(sample.input, activation);
  
  const array_t& y = activation.back();

  array_t error;
  error.resize(y.size());
  std::transform(y.begin(), y.end(), sample.output.begin(), error.begin(), std::bind(&Loss::error, _loss, _1, _2));
  
  size_t layerIdx = _network.size();
  while(layerIdx-- > 0)
  {
    const array_t& input = (layerIdx == 0) ? sample.input : activation[layerIdx - 1];
    updateLayer(layerIdx, input, activation[layerIdx], error);
  }
}

bool StochasticGradientDescent::train(const sample_list_t& samples, size_t maxIterations, double epsilon)
{
  initNeuronData();

  bool success(false);

  //auto sampleIdx = std::bind(std::uniform_int_distribution<size_t>(0, samples.size() - 1), std::default_random_engine());

  for(size_t i=0; i<maxIterations; ++i)
  {
    //learnSample(samples[sampleIdx()]);
    for(size_t j=0; j<samples.size(); ++j) 
    {
      // learnSample(samples[j]);
      learnSample2(samples[j]);
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
  return std::make_unique<SgdNeuronData>(neuron);
}

SgdNeuronData& StochasticGradientDescent::neuronData(size_t layerIdx, size_t neuronIdx)
{
  return static_cast<SgdNeuronData&>(*(_neuronData.at(layerIdx).at(neuronIdx)));
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