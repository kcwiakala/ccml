#ifndef CCML_NEURON_LAYER_HPP
#define CCML_NEURON_LAYER_HPP

#include <functional>

#include <Neuron.hpp>

#include "TypedLayer.hpp"

namespace ccml {

class NeuronLayer: public TypedLayer
{
public:
  typedef std::function<void(Neuron&, size_t)> neuron_updater_t;

  typedef std::function<void(const Neuron&, size_t)> neuron_reader_t;

public:
  virtual size_t outputSize() const;

  void forEachNeuron(const neuron_updater_t& updater);

  void forEachNeuron(const neuron_reader_t& reader) const;

  void init(const Initializer& weightInit, const Initializer& biasInit);

  virtual void toStream(std::ostream& stream) const;

protected:
  NeuronLayer(const std::string& type, size_t layerSize, size_t neuronSize, const Activation& activation);

protected:
  typedef std::vector<Neuron> neuron_list_t;
  neuron_list_t _neurons;
};

} // namespace ccml

#endif // CCML_NEURON_LAYER_HPP