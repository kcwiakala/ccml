#include <algorithm>
#include <iostream> 

#include "NeuronLayer.hpp"

#include <Utils.hpp>

namespace ccml {

using namespace std::placeholders;

NeuronLayer::~NeuronLayer() = default;

void NeuronLayer::error(const array_t& y, const array_t& dy, array_t& e) const
{
  transfer().deriverate(y, e);
  std::transform(e.cbegin(), e.cend(), dy.cbegin(), e.begin(), [](value_t ei, value_t dyi) {
    return dyi * ei;
  });
}

void NeuronLayer::init(const initializer_t& weightInit, const initializer_t& biasInit)
{
  std::for_each(_nodes.begin(), _nodes.end(), [&](Node& node) {
    node.init(weightInit, biasInit);
  });
}

size_t NeuronLayer::size() const
{
  return _nodes.size();
}

const Node& NeuronLayer::node(size_t idx) const
{
  return _nodes[idx];
}

Node& NeuronLayer::node(size_t idx)
{
  return _nodes[idx];
}

void NeuronLayer::toStream(std::ostream& stream) const
{
  stream << _type << ":{is:" << inputSize() << ",os:" << outputSize();
  stream << ",n:[";
  for(size_t i=0; i<_nodes.size(); ++i) 
  {
    stream << ((i>0) ? "," : "") << _nodes[i];
  }
  stream << "],t:" << transfer().name() << "}";
}

} // namespace ccml