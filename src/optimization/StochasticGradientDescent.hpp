#ifndef CCML_STOCHASTIC_GRADIENT_DESCENT_HPP
#define CCML_STOCHASTIC_GRADIENT_DESCENT_HPP

#include <Loss.hpp>
#include <Network.hpp>
#include <Sample.hpp>

namespace ccml {

class StochasticGradientDescent 
{
public:
  StochasticGradientDescent(loss_ptr_t loss, double rate);

  void learnSample(Network& network, const Sample& sample) const;

  bool train(Network& network, const sample_list_t& samples, size_t maxIterations, double epsilon) const;

protected:
  void updateLayer(Network& network, size_t layerIdx, const array_t& input, const array_t& output, array_t& error) const;

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
  loss_ptr_t _loss;
  double _rate;
};

} // namespace ccml

#endif // CCML_STOCHASTIC_GRADIENT_DESCENT_HPP