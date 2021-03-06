#ifndef CCML_NEURON_LAYER_HPP
#define CCML_NEURON_LAYER_HPP

#include <functional>

#include <Node.hpp>
#include <transfer/Transfer.hpp>
#include "TransferLayer.hpp"

namespace ccml {

class NeuronLayer: public TransferLayer
{
public:
  typedef std::function<void(const array_t&, size_t)> error_reader_t;

public:
  virtual ~NeuronLayer();

  virtual void splitError(const array_t& x, const array_t& error, const error_reader_t& reader) const = 0;

  void init(const initializer_t& weightInit, const initializer_t& biasInit);

  virtual void toStream(std::ostream& stream) const;

public:
  size_t size() const;

  const Node& node(size_t idx) const;

  Node& node(size_t idx);

protected:
  template<typename Tp, typename Tr>
  NeuronLayer(Tp&& type, size_t layerSize, size_t neuronSize, Tr&& transfer):
    TransferLayer(layerSize, std::forward<Tr>(transfer)),
    _nodes(layerSize, Node(neuronSize)),
    _type(std::forward<Tp>(type))
  {}

protected:
  using node_list_t = std::vector<Node>;
  node_list_t _nodes;
  const std::string _type;
};

typedef std::shared_ptr<NeuronLayer> neuron_layer_ptr_t;

} // namespace ccml

#endif // CCML_NEURON_LAYER_HPP