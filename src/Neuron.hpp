#ifndef CCML_NEURON_HPP
#define CCML_NEURON_HPP

#include <vector>

#include <Activation.hpp>
#include <Initialization.hpp>
#include <Types.hpp>

namespace ccml {

class Neuron
{
public:
  Neuron(size_t inputSize, const Activation& activation);

public:
  size_t size() const;

  void init(const Initializer& weightInit, const Initializer& biasInit);

  void init(const Initializer& initializer);

  value_t output(const array_t& input) const;

  void adjust(const array_t& deltaWeight, value_t deltaBias);

private:
  double net(const array_t& input) const;

private:
  const Activation _activation;
  value_t _bias;
  array_t _weights;
};

} // namespace ccml

#endif