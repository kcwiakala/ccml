#include <functional>

namespace ccml {


class Initializer 
{
public:
  typedef std::function<double()> generator_type;

  Initializer(const generator_type&& generator): _generator(generator) {}

  static Initializer constant(double value);

  static Initializer uniform(double min, double max);  

  static Initializer normal(double mean, double sigma);

  generator_type::result_type operator()() const
  {
    return _generator();
  }

private:
  generator_type _generator;
};

} // namespace ccml