#ifndef CCML_NEURON_HPP
#define CCML_NEURON_HPP

#include <vector>

#include <Activation.hpp>
#include <Initialization.hpp>

namespace ccml {

class Neuron
{
public:
  Neuron(size_t inputSize, const Activation& activation);

public:
  size_t size() const;

  void init(const Initializer& weightInit, const Initializer& biasInit);

  void init(const Initializer& initializer);

  double output(const std::vector<double>& input) const;

  void adjust(const std::vector<double>& deltaWeight, double deltaBias);

private:
  double net(const std::vector<double>& input) const;

private:
  const Activation _activation;
  double _bias;
  std::vector<double> _weights;
};

} // namespace ccml

#endif