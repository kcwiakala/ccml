#ifndef CCML_NEURON_HPP
#define CCML_NEURON_HPP

#include <memory>
#include <ostream>

#include <Activation.hpp>
#include <Initialization.hpp>
#include <Serializable.hpp>
#include <Types.hpp>

namespace ccml {

class Neuron: public Serializable
{
public:
  Neuron(size_t inputSize, const Activation& activation);

public:
  size_t size() const;

  void init(const Initializer& weightInit, const Initializer& biasInit);

  value_t output(const array_t& input) const;

  void adjust(const array_t& deltaWeight, value_t deltaBias);

  const array_t& weights() const;

  const value_t& bias() const;

  virtual void toStream(std::ostream& stream) const;

private:
  double net(const array_t& input) const;

private:
  const Activation& _activation;
  value_t _bias;
  array_t _weights;
};

inline size_t Neuron::size() const
{
  return _weights.size();
}

inline const array_t& Neuron::weights() const
{
  return _weights;
}

inline const value_t& Neuron::bias() const
{
  return _bias;
}

} // namespace ccml

#endif