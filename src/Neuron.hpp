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
  void init(const Initializer& weightInit, const Initializer& biasInit);

  void init(const Initializer& initializer);

  double output(const std::vector<double>& input) const;

private:
  double net(const std::vector<double>& input) const;

private:
  const Activation _activation;
  double _bias;
  std::vector<double> _weights;
};

} // namespace ccml

#endif