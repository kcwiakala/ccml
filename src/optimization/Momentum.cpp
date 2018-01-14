#include <Utils.hpp>
#include <optimization/Momentum.hpp>

namespace ccml {

Momentum::Momentum(Network& network, loss_ptr_t loss, double rate, double momentum):
  StochasticGradientDescent(network, loss, rate),
  _momentum(momentum)
{
    _neuronData.resize(_network.size());
    for(size_t i=0; i<network.size(); ++i) 
    {
      layer_ptr_t layer = network.layer(i);
      _neuronData[i].resize(layer->outputSize());
      neuron_layer_ptr_t neuronLayer = std::dynamic_pointer_cast<NeuronLayer>(layer);
      if(neuronLayer)
      {
        neuronLayer->forEachNeuron([&](const Neuron& neuron, size_t j) {
          _neuronData[i][j].deltaBias = 0.0;
          _neuronData[i][j].deltaWeight.assign(neuron.weights().size(), 0.0);
        });
      }
    }
}

Momentum::~Momentum()
{
}

void Momentum::adjust(neuron_layer_ptr_t layer, const array_t& input, const array_t& error, size_t layerIndex)
{
  auto& layerData = _neuronData[layerIndex];
  
  layer->adjust(input, error, [&](Neuron& n, const array_t& wg, size_t i) {
    NeuronData& neuronData = layerData[i];
    array_t& dw = neuronData.deltaWeight;
    std::transform(wg.begin(), wg.end(), dw.begin(), dw.begin(), [&](value_t wgi, value_t dwi) {
      return wgi * _rate + dwi * _momentum;
    });
    neuronData.deltaBias = error[i] * _rate + neuronData.deltaBias * _momentum;
    n.adjust(neuronData.deltaWeight, neuronData.deltaBias);
  });
}

void Momentum::reset()
{
  for_each(_neuronData, [](NeuronData& neuronData) {
    neuronData.deltaBias = 0.0;
    neuronData.deltaWeight.assign(neuronData.deltaWeight.size(), 0.0);
  });
}

} // namespace ccml