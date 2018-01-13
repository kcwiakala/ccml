#ifndef CCML_STOCHASTIC_GRADIENT_DESCENT_HPP
#define CCML_STOCHASTIC_GRADIENT_DESCENT_HPP

#include <Network.hpp>
#include <Sample.hpp>

namespace ccml {

class Loss
{

};

class StochasticGradientDescent 
{
public:
  StochasticGradientDescent(const Loss& loss, double rate);

  void learnSample(Network& network, const Sample& sample);

  bool train(Network& network, const sample_list_t& samples, size_t maxIterations, double epsilon) const;

protected:
  void updateLayer(Network& network, size_t layerIdx, const array_2d_t& activation, array_t& error);

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

private:
  Loss _loss;
  double _rate;
};

} // namespace ccml

#endif // CCML_STOCHASTIC_GRADIENT_DESCENT_HPP