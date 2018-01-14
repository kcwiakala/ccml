#ifndef CCML_MOMENTUM_HPP
#define CCML_MOMENTUM_HPP

#include <optimization/StochasticGradientDescent.hpp>

namespace ccml {

class Momentum: public StochasticGradientDescent
{
public:
  Momentum(Network& network, loss_ptr_t loss, double rate, double momentum);

  virtual ~Momentum();

protected:
  virtual void adjust(neuron_layer_ptr_t layer, const array_t& input, const array_t& error, size_t layerIndex);

  virtual void reset();

private:
  struct NeuronData
  {
    array_t deltaWeight;
    value_t deltaBias;
  };

  vector_2d<NeuronData> _neuronData;
  const double _momentum;
};

} // namespace ccml

#endif // CCML_MOMENTUM_HPP