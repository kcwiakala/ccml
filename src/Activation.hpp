#include <functional>

namespace ccml {

class Activation 
{
public:
  double operator()(double x) const
  {
    return _fn(x);
  }

  double derivative(double x, double y) const
  {
    return _df(x);
  }

  static Activation& sigmoid();

private:
  typedef std::function<double(double)> function_t;

  Activation(const function_t& fn, const function_t& df): _fn(fn), _df(df) {}

private:
  const function_t _fn;
  const function_t _df;
};

} // namespace ccml