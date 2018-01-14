#ifndef CCML_STOCHASTIC_GRADIENT_DESCENT_HPP
#define CCML_STOCHASTIC_GRADIENT_DESCENT_HPP

#include <Loss.hpp>
#include <Network.hpp>
#include <layer/NeuronLayer.hpp>
#include <Sample.hpp>

#include <optimization/NeuronData.hpp>


namespace ccml {

class SgdNeuronData: public NeuronData
{
public:
  SgdNeuronData(const Neuron& neuron);

  virtual void reset();

  array_t batchWeightGradient;
  value_t batchBiasGradient;
};

class StochasticGradientDescent 
{
public:
  StochasticGradientDescent(Network& network, loss_ptr_t loss, double rate);

  virtual ~StochasticGradientDescent();

  bool train(const sample_list_t& samples, size_t maxIterations, double epsilon);

protected:
  void learnSample(const Sample& sample);

  void updateLayer(size_t layerIdx, const array_t& input, const array_t& output, array_t& error);

protected:
  virtual void adjust(neuron_layer_ptr_t layer, const array_t& input, const array_t& error, size_t layerIndex);

  virtual neuron_data_ptr_t createNeuronData(const Neuron& neuron) const;

  virtual SgdNeuronData& neuronData(size_t layerIdx, size_t neuronIdx);

  virtual void initNeuronData();

protected:
  Network& _network;
  loss_ptr_t _loss;
  const double _rate;
  vector_2d<neuron_data_ptr_t> _neuronData;
};

} // namespace ccml

#endif // CCML_STOCHASTIC_GRADIENT_DESCENT_HPP