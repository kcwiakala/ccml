#ifndef CCML_ACTIVATION_HPP
#define CCML_ACTIVATION_HPP

#include <functional>

namespace ccml {

class Activation
{
public:
  double operator()(double x) const
  {
    return _fn(x);
  }

  double derivative(double y) const
  {
    return _df(y);
  }

protected:
  typedef std::function<double(double)> function_t;

  Activation(const function_t& fn, const function_t& df): _fn(fn), _df(df) {}

private:
  const function_t _fn;
  const function_t _df;
};

} // namespace ccml

#endif