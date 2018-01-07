#ifndef CCML_INITIALIZER_HPP
#define CCML_INITIALIZER_HPP

#include <functional>

namespace ccml {

class Initializer 
{
public:
  typedef std::function<double()> generator_t;

  Initializer(const double value);

  Initializer(const generator_t&& generator): _generator(generator) {}

  static Initializer constant(double value);

  static Initializer uniform(double min, double max);  

  static Initializer normal(double mean, double sigma);

  generator_t::result_type operator()() const
  {
    return _generator();
  }

private:
  generator_t _generator;
};

} // namespace ccml

#endif 