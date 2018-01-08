#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>

#include <Perceptron.hpp>

namespace ccml {
  
Perceptron::Perceptron(size_t inputSize):
  _neuron(inputSize, ccml::Activation::heaviside())
{
}

void Perceptron::init(const Initializer& weightInit, const Initializer& biasInit)
{
  _neuron.init(weightInit, biasInit);
}

void Perceptron::init(const Initializer& initializer)
{
  init(initializer, initializer);
}

double Perceptron::output(const std::vector<double>& input) const
{
  return _neuron.output(input);
}

double Perceptron::error(const Sample& sample) const
{
  assert(sample.output.size() == 1);
  return sample.output[0] - output(sample.input);
}

double Perceptron::loss(const Sample& sample) const
{
  return std::fabs(error(sample));
}

double Perceptron::loss(const sample_list_t& samples) const
{
  return std::accumulate(samples.begin(), samples.end(), 0.0, [&](double sum, const Sample& sample) {
    return sum + loss(sample);
  });
}

void Perceptron::adjust(const input_t& input, double error, input_t& aux) 
{
  std::transform(input.begin(), input.end(), aux.begin(), [=](double x) {
    return x * error;
  });
  _neuron.adjust(aux, error);
}

bool Perceptron::learn(const sample_list_t& samples, double minLoss, size_t maxIterations)
{
  size_t iter(0);
  input_t aux(_neuron.size()); 

  while((loss(samples) > minLoss) && (++iter < maxIterations)) 
  {
    std::for_each(samples.begin(), samples.end(), [&](const Sample& sample) {
      adjust(sample.input, error(sample), aux);
    });
  }
  return iter < maxIterations;
}

} // namespace ccml