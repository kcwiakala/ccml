#include <random>

#include "Initialization.hpp"

namespace ccml {

Initializer::Initializer(const double value): 
  _generator([=](){
    return value;
  })
{
}

Initializer Initializer::constant(double value)
{
  return Initializer(value);
}

Initializer Initializer::uniform(double min, double max)
{
  auto generator = std::bind(std::uniform_real_distribution<double>(min, max), std::default_random_engine());
  return Initializer(generator);
}

Initializer Initializer::normal(double mean, double sigma)
{
  auto generator = std::bind(std::normal_distribution<double>(mean, sigma), std::default_random_engine());
  return Initializer(generator);
}

} // namespace ccml