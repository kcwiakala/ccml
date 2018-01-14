#ifndef CCML_STOCHASTIC_GRADIENT_DESCENT_HPP
#define CCML_STOCHASTIC_GRADIENT_DESCENT_HPP

#include <Loss.hpp>
#include <Network.hpp>
#include <layer/NeuronLayer.hpp>
#include <Sample.hpp>

namespace ccml {

class StochasticGradientDescent 
{
public:
  StochasticGradientDescent(Network& network, loss_ptr_t loss, double rate);

  virtual ~StochasticGradientDescent();

  bool train(const sample_list_t& samples, size_t maxIterations, double epsilon);

protected:
  void learnSample(const Sample& sample);

  void updateLayer(size_t layerIdx, const array_t& input, const array_t& output, array_t& error);

  virtual void adjust(neuron_layer_ptr_t layer, const array_t& input, const array_t& error, size_t layerIndex);

  virtual void reset();

  template<typename T>
  void prepareNeuronData(const Network& network, vector_2d<T>& neuronData)
  {
    neuronData.clear();
    neuronData.resize(network.size());
    for(size_t i=0; i<network.size(); ++i) 
    {
      neuronData[i].resize(network.layer(i)->outputSize());
    }
  } 

protected:
  Network& _network;
  loss_ptr_t _loss;
  const double _rate;
};

} // namespace ccml

#endif // CCML_STOCHASTIC_GRADIENT_DESCENT_HPP