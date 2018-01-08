#ifndef CCML_PERCEPTRON_HPP
#define CCML_PERCEPTRON_HPP

#include <Neuron.hpp>
#include <Sample.hpp>

namespace ccml {

class Perceptron
{
public:
  Perceptron(size_t inputSize);

  void init(const Initializer& weightInit, const Initializer& biasInit);

  void init(const Initializer& initializer);

  double output(const std::vector<double>& input) const;

  double error(const Sample& sample) const;

  double loss(const Sample& sample) const;

  double loss(const sample_list_t& samples) const;

  bool learn(const sample_list_t& samples, double minLoss = 0.001, size_t maxIterations = 10000);

private:
  void adjust(const input_t& input, double error, input_t& aux);

private:
  Neuron _neuron;
};

} // namespace ccml

#endif // CCML_PERCEPTRON_HPP