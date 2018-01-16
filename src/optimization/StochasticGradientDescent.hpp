#ifndef CCML_STOCHASTIC_GRADIENT_DESCENT_HPP
#define CCML_STOCHASTIC_GRADIENT_DESCENT_HPP

#include <Loss.hpp>
#include <Network.hpp>
#include <layer/NeuronLayer.hpp>
#include <Sample.hpp>

#include <optimization/NeuronData.hpp>

#include "Backpropagation.hpp"

namespace ccml {

struct GradientData
{
  GradientData(const Neuron& neuron);

  void reset();

  array_t weights;
  value_t bias;
};

class StochasticGradientDescent: public Backpropagation
{
public:
  StochasticGradientDescent(Network& network, const loss_ptr_t& loss, double rate);

  virtual ~StochasticGradientDescent();

  bool train(const sample_list_t& samples, size_t maxIterations, double epsilon);

protected:
  void learnSample(const Sample& sample);

  void updateGradients(const array_t& input, const array_2d_t& activation, const array_2d_t& error);

  void adjustNeurons();

protected:
  virtual void adjustNeuron(Neuron& neuron, GradientData& gradients, size_t layerIdx, size_t neuronIdx);

  virtual neuron_data_ptr_t createNeuronData(const Neuron& neuron) const;

  virtual NeuronData& neuronData(size_t layerIdx, size_t neuronIdx);

  virtual void initNeuronData();

protected:
  const double _rate;
  vector_2d<neuron_data_ptr_t> _neuronData;
  vector_2d<GradientData> _gradients;
};

} // namespace ccml

#endif // CCML_STOCHASTIC_GRADIENT_DESCENT_HPP