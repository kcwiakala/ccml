#include <algorithm>
#include <functional>
#include <iostream>
#include <random>

#include <layer/NeuronLayer.hpp>
#include <Utils.hpp>
#include "StochasticGradientDescent.hpp"

namespace ccml {

using namespace std::placeholders;

StochasticGradientDescent::StochasticGradientDescent(loss_ptr_t loss, double rate):
  _loss(loss), _rate(rate)
{
}

void StochasticGradientDescent::updateLayer(Network& network, size_t layerIdx, 
  const array_t& input, const array_t& output, array_t& error) const
{
  thread_local static array_t aux;
  thread_local static array_t backpropagatedError;
  
  layer_ptr_t layer = network.layer(layerIdx);

  neuron_layer_ptr_t neuronLayer = std::dynamic_pointer_cast<NeuronLayer>(layer);
  if(neuronLayer)
  {
    // Calculate error gradient
    neuronLayer->error(output, error, aux);
    error.swap(aux);

    if(layerIdx > 0) 
    {
      neuronLayer->backpropagate(error, backpropagatedError);  
    }

    // Adjust neurons in the layer
    neuronLayer->adjust(input, error, [&](Neuron& n, const array_t& wg, size_t i) {
      aux.resize(wg.size());
      std::transform(wg.begin(), wg.end(), aux.begin(), [=](value_t wgi) {
        return wgi * _rate;
      });
      n.adjust(aux, error[i] * _rate);
    });

    error.swap(backpropagatedError);
  }
  else if(layerIdx > 0)
  {
    layer->backpropagate(error, backpropagatedError);
    error.swap(backpropagatedError);
  }
}

void StochasticGradientDescent::learnSample(Network& network, const Sample& sample) const
{
  thread_local static array_t aux;

  //assert((network.inputSize() != sample.input.size()) || (network.outputSize() != sample.output.size()))
  
  array_2d_t activation;
  network.output(sample.input, activation);
  
  const array_t& y = activation.back();
  aux.resize(y.size());
  std::transform(y.begin(), y.end(), sample.output.begin(), aux.begin(), std::bind(&Loss::error, _loss, _1, _2));
  
  size_t layerIdx = network.size();
  while(layerIdx-- > 0)
  {
    const array_t& input = (layerIdx == 0) ? sample.input : activation[layerIdx - 1];
    updateLayer(network, layerIdx,  input, activation[layerIdx], aux);
  }
}

bool StochasticGradientDescent::train(Network& network, const sample_list_t& samples, size_t maxIterations, double epsilon) const
{
  bool success(false);
  //auto sampleIdx = std::bind(std::uniform_int_distribution<size_t>(0, samples.size() - 1), std::default_random_engine());

  for(size_t i=0; i<maxIterations; ++i)
  {
    //learnSample(network, samples[sampleIdx()]);
    for(size_t j=0; j<samples.size(); ++j) 
    {
      learnSample(network, samples[j]);
    }

    const double totalLoss = _loss->compute(network, samples);
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

} // namespace ccml