#include <random>

#include "Initialization.hpp"

namespace ccml {
namespace initializer {
  
namespace {

std::random_device randomDevice;
std::default_random_engine randomEngine(randomDevice());

} // namespace

initializer_t constant(value_t value)
{
  return [=]() { 
    return value; 
  };
}

initializer_t uniform(value_t min, value_t max)
{
  return std::bind(std::uniform_real_distribution<value_t>(min, max), std::ref(randomEngine));
}

initializer_t normal(value_t mean, value_t sigma)
{
  return std::bind(std::normal_distribution<value_t>(mean, sigma), std::ref(randomEngine));
}

} // namespace initializer
} // namespace ccml