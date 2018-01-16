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

  typedef std::function<void(Neuron&, const array_t&, size_t)> neuron_adjuster_t;

public:
  virtual ~NeuronLayer() {}

  virtual size_t outputSize() const;

  virtual void error(const array_t& y, const array_t& dy, array_t& e) const;

  virtual void splitError(const array_t& x, const array_t& error, const neuron_adjuster_t& adjuster);

  void forEachNeuron(const neuron_updater_t& updater);

  void forEachNeuron(const neuron_reader_t& reader) const;

  void init(const Initializer& weightInit, const Initializer& biasInit);

  virtual void toStream(std::ostream& stream) const;

public:
  size_t size() const;

  const Neuron& neuron(size_t idx) const;

  Neuron& neuron(size_t idx);

protected:
  NeuronLayer(const std::string& type, size_t layerSize, size_t neuronSize, const Activation& activation);

protected:
  typedef std::vector<Neuron> neuron_list_t;
  neuron_list_t _neurons;

  const Activation& _activation;
};

typedef std::shared_ptr<NeuronLayer> neuron_layer_ptr_t;

} // namespace ccml

#endif // CCML_NEURON_LAYER_HPP