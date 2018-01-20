#ifndef CCML_NEURON_HPP
#define CCML_NEURON_HPP

#include <Node.hpp>
#include <Transfer.hpp>

namespace ccml {

class Neuron: public Node
{
public:
  Neuron(size_t inputSize, const Transfer& transfer);

public:
  value_t output(const array_t& input) const;

  virtual void toStream(std::ostream& stream) const;

private:
  const Transfer _transfer;
};

} // namespace ccml

#endif