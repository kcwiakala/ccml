#include <algorithm>
#include <random>

#include <layer/NeuronLayer.hpp>
#include <Utils.hpp>
#include "StochasticGradientDescent.hpp"

namespace ccml {

struct SgdNeuronData
{
  value_t gradient;
};

void StochasticGradientDescent::updateLayer(Network& network, size_t layerIdx, const array_2d_t& activation, array_t& error)
{
  thread_local static array_t aux;
  layer_ptr_t layer = network.layer(layerIdx);

  // If we are in hidden layer, backpropagate error through next layer
  if(layerIdx < network.size() - 1) 
  {
    network.layer(layerIdx + 1)->backpropagate(error, aux);
    error.swap(aux);
  }

  neuron_layer_ptr_t neuronLayer = std::dynamic_pointer_cast<NeuronLayer>(layer);
  if(neuronLayer)
  {
    // Calculate error gradient
    neuronLayer->error(activation[layerIdx], error, aux);
    error.swap(aux);
    
    // Adjust neurons in the layer
    const array_t& layerInput = activation[layerIdx - 1];
    neuronLayer->adjust(layerInput, error, [&](Neuron& n, const array_t& wg, size_t i) {
      aux.resize(wg.size());
      std::transform(wg.begin(), wg.end(), aux.begin(), [=](value_t wgi) {
        return wgi * _rate;
      });
      n.adjust(aux, error[i]);
    });
  }
}

void StochasticGradientDescent::learnSample(Network& network, const Sample& sample)
{
  thread_local static array_t aux;

  if((network.inputSize() != sample.input.size()) || (network.outputSize() != sample.output.size()))
  {
    // ERROR
  }
  array_2d_t activation;
  network.output(sample.input, activation);
  
  const array_t& y = activation.back();
  aux.resize(y.size());
  std::transform(y.begin(), y.end(), sample.output.begin(), aux.begin(), [](value_t output, value_t expected) {
    return expected - output;
  });

  size_t layerIdx = network.size() - 1;
  while(layerIdx > 0)
  {
    updateLayer(network, layerIdx--, activation, aux);
  }
}

bool StochasticGradientDescent::train(Network& network, const sample_list_t& samples, size_t maxIterations, double epsilon) const
{
  // bool success(false);
  // auto sampleIdx = std::bind(std::uniform_int_distribution<size_t>(0, samples.size() - 1), std::default_random_engine());

  // for(size_t i=0; i<maxIterations; ++i)
  // {
  //   learnSample(network, samples[sampleIdx()]);
  //   const double totalLoss = _loss.total(network, samples);
  //   if((i % debugIteration) == 0)
  //   {
  //     std::cout << "Total loss after " << i << " iterations is: " << totalLoss;
  //   }
  //   if(totalLoss < epsilon)
  //   {
  //     success = true;
  //     break;
  //   }
  // }

  // return success;
}

} // namespace ccml