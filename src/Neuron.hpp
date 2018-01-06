#include <vector>

#include <Activation.hpp>
#include <Initialization.hpp>

namespace ccml {

class Neuron
{
public:
  Neuron(size_t inputSize, Activation& activation);

  Neuron(const Neuron&) = delete;
  Neuron& operator=(const Neuron&) = delete;

  Neuron(Neuron&&) = default;
  Neuron& operator=(Neuron&&) = default;

public:
  void init(const Initializer& weightInit, const Initializer& biasInit);

  void init(const Initializer& initializer);

  double output(const std::vector<double>& input) const;

  double activate(const std::vector<double>& input);

  double errorGradient(double error) const;

private:
  double net(const std::vector<double>& input) const;

private:
  Activation& _activation;
  double _net;
  double _value;
  double _bias;
  std::vector<double> _weights;
};

} // namespace ccml