#include <algorithm>
#include <functional>
#include <future>
#include <iostream>
#include <iterator>
#include <random>

#include <layer/NeuronLayer.hpp>
#include "StochasticGradientDescent.hpp"

namespace ccml {
namespace {

constexpr size_t BATCH_SIZE_THRESHOLD = 10;

} // namespace

GradientData::GradientData(const Node& node):
  weights(node.size(), 0.0),
  bias(0.0)
{
}

void GradientData::reset()
{
  bias = 0.0;
  std::fill(weights.begin(), weights.end(), 0.0);
}

StochasticGradientDescent::StochasticGradientDescent(Network& network, loss_ptr_t loss, double rate):
    Backpropagation(network, std::move(loss)),
    _rate(rate),_tp(std::thread::hardware_concurrency())
{
  _gradients.resize(_network.size());

  for(size_t layerIdx=0; layerIdx < _network.size(); ++layerIdx)
  {
    neuron_layer_ptr_t layer = _network.neuronLayer(layerIdx);
    if(layer)
    {
      _gradients[layerIdx].reserve(layer->size());
      for(size_t nodeIdx=0; nodeIdx < layer->size(); ++nodeIdx)
      {
        _gradients[layerIdx].emplace_back(GradientData(layer->node(nodeIdx)));
      }
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
      layer->splitError(layerInput, error[layerIdx], [this, layerIdx, &error](const array_t& inputError, size_t nodeIdx) {
        std::lock_guard<std::mutex> lock(_mtx);
        GradientData& data = _gradients[layerIdx][nodeIdx];
        std::transform(data.weights.cbegin(), data.weights.cend(), 
            inputError.cbegin(), data.weights.begin(), 
            std::minus<value_t>());
        data.bias -= error[layerIdx][nodeIdx];
      });
    }
  }
}

void StochasticGradientDescent::normalizeGradients(size_t batchSize)
{
  for(size_t layerIdx=0; layerIdx < _network.size(); ++layerIdx)
  {
    neuron_layer_ptr_t layer = _network.neuronLayer(layerIdx);
    if(layer)
    {
      for(size_t nodeIdx=0; nodeIdx < layer->size(); ++nodeIdx)
      {
        GradientData& gradients = _gradients[layerIdx][nodeIdx];
        std::transform(gradients.weights.cbegin(), gradients.weights.cend(), gradients.weights.begin(), [batchSize](value_t g) {
          return g / batchSize;
        });
        gradients.bias /= batchSize;
      }
    }
  }
}

void StochasticGradientDescent::adjustNodes()
{
  for(size_t layerIdx=0; layerIdx < _network.size(); ++layerIdx)
  {
    neuron_layer_ptr_t layer = _network.neuronLayer(layerIdx);
    if(layer)
    {
      for(size_t nodeIdx=0; nodeIdx < layer->size(); ++nodeIdx)
      {
        GradientData& gradients = _gradients[layerIdx][nodeIdx];
        adjustNode(layer->node(nodeIdx), gradients, layerIdx, nodeIdx);
        gradients.reset();
      }
    }
  }
}

void StochasticGradientDescent::adjustNode(Node& node, GradientData& gradients, size_t layerIdx, size_t nodeIdx)
{
  array_t& wg = gradients.weights;
  std::transform(wg.cbegin(), wg.cend(), wg.begin(), [this](value_t wgi) {
    return wgi * _rate;
  });
  node.adjust(wg, gradients.bias * _rate);
}

void StochasticGradientDescent::learnSample(const Sample& sample)
{
  thread_local array_2d_t activation;
  thread_local array_2d_t error;

  // Feed forward and backpropagate error
  passSample(sample, activation, error);

  // Update gradients for each neuron in the network
  updateGradients(sample.input, activation, error);
}

void StochasticGradientDescent::learnBatch(const sample_batch_t& batch)
{  
  const size_t batchSize = std::distance(batch.first, batch.second);
  if(batchSize >= BATCH_SIZE_THRESHOLD)
  {
    std::vector<std::future<void>> futures;
    futures.reserve(std::distance(batch.first, batch.second));
    std::for_each(batch.first, batch.second, [this, &futures](auto& sample) {
      futures.emplace_back(_tp.enqueue([this, &sample](){
        learnSample(sample);
      }));
    });
    for(auto& fut: futures)
    {
      fut.wait();
    }
  }
  else
  {
    std::for_each(batch.first, batch.second, [this](auto& sample) {
      learnSample(sample);
    });
  }

  normalizeGradients(std::distance(batch.first, batch.second));
  adjustNodes();
}

} // namespace ccml