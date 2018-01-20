#ifndef CCML_NODE_HPP
#define CCML_NODE_HPP

#include <Initialization.hpp>
#include <Serializable.hpp>
#include <Types.hpp>

namespace ccml {

class Node: public Serializable
{
public:
  Node(size_t inputSize);

  void init(const initializer_t& weightInit, const initializer_t& biasInit);

  value_t output(const array_t& input) const;

  void adjust(const array_t& deltaWeight, value_t deltaBias);

  virtual void toStream(std::ostream& stream) const;

public:
  size_t size() const
  {
    return _weights.size();
  }

  const array_t& weights() const
  {
    return _weights;
  }

  value_t bias() const
  {
    return _bias;
  }

protected:
  value_t _bias;
  array_t _weights;
};

} // namespace ccml

#endif // CCML_NODE_HPP