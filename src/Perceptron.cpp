#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>

#include <Perceptron.hpp>
#include <transfer/Heaviside.hpp>

namespace ccml {
  
Perceptron::Perceptron(size_t inputSize):
  _layer(inputSize, 1, std::make_shared<transfer::Heaviside>())
{
}

void Perceptron::init(const initializer_t& weightInit, const initializer_t& biasInit)
{
  _layer.init(weightInit, biasInit);
}

void Perceptron::init(const initializer_t& initializer)
{
  init(initializer, initializer);
}

value_t Perceptron::output(const array_t& input) const
{
  static array_t aux;
  _layer.output(input, aux);
  return aux[0];
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

void Perceptron::adjust(const array_t& input, double error, array_t& aux) 
{
  std::transform(input.begin(), input.end(), aux.begin(), [=](value_t x) {
    return x * error;
  });
  _layer.node(0).adjust(aux, error);
}

bool Perceptron::learn(const sample_list_t& samples, double minLoss, size_t maxIterations)
{
  size_t iter(0);
  array_t aux(_layer.inputSize()); 

  while((loss(samples) > minLoss) && (++iter < maxIterations)) 
  {
    std::for_each(samples.begin(), samples.end(), [&](const Sample& sample) {
      adjust(sample.input, error(sample), aux);
    });
  }
  return iter < maxIterations;
}

} // namespace ccml