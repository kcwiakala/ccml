#ifndef CCML_NEURON_LAYER_HPP
#define CCML_NEURON_LAYER_HPP

#include <Neuron.hpp>

#include "AbstractLayer.hpp"

namespace ccml {

class NeuronLayer: public AbstractLayer
{
public:
  virtual size_t outputSize() const;

protected:
  NeuronLayer(size_t layerSize, size_t neuronSize, const Activation& activation);

protected:
  typedef std::vector<Neuron> neuron_list_t;
  neuron_list_t _neurons;
};

} // namespace ccml

#endif // CCML_NEURON_LAYER_HPP