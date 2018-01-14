#include <algorithm>
#include <functional>
#include <iostream>
#include <random>

#include <layer/NeuronLayer.hpp>
#include <Utils.hpp>
#include "StochasticGradientDescent.hpp"

namespace ccml {

using namespace std::placeholders;

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

  layer->adjust(input, error, [&](Neuron& n, const array_t& wg, size_t i) {
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

  layer_ptr_t layer = _network.layer(layerIdx);
  
  neuron_layer_ptr_t neuronLayer = std::dynamic_pointer_cast<NeuronLayer>(layer);
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
    // Calculate backpropagated error and return it for next layers
    layer->backpropagate(error, aux);
    error.swap(aux);
  }
}

void StochasticGradientDescent::learnSample(const Sample& sample)
{
  thread_local static array_t error;
  thread_local static array_2d_t activation;

  //assert((network.inputSize() != sample.input.size()) || (network.outputSize() != sample.output.size()))
  _network.output(sample.input, activation);
  
  const array_t& y = activation.back();
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
  reset();

  bool success(false);

  //auto sampleIdx = std::bind(std::uniform_int_distribution<size_t>(0, samples.size() - 1), std::default_random_engine());

  for(size_t i=0; i<maxIterations; ++i)
  {
    //learnSample(network, samples[sampleIdx()]);
    for(size_t j=0; j<samples.size(); ++j) 
    {
      learnSample(samples[j]);
    }

    const double totalLoss = _loss->compute(_network, samples);
    // std::cout << "Total loss after " << i << " iterations is: " << totalLoss << std::endl;
    // if((i % debugIteration) == 0)
    // {
    // }
    if(totalLoss < epsilon)
    {
      success = true;
      break;
    }
  }

  return success;
}

void StochasticGradientDescent::reset()
{
  // empty
}

} // namespace ccml