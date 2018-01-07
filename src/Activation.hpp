#ifndef CCML_ACTIVATION_HPP
#define CCML_ACTIVATION_HPP

#include <functional>
#include <set>
#include <string>

namespace ccml {

class Activation
{
public:
  const std::string& name()
  {
    return _name;
  }

  double operator()(double x) const
  {
    return _fn(x);
  }

  double derivative(double y) const
  {
    return _df(y);
  }

public:
  static const Activation& heaviside();

  static const Activation& relu();

  static const Activation& sigmoid();

public:
  bool operator<(const Activation& other) const
  {
    return _name < other._name;
  }

  bool operator==(const Activation& other) const
  {
    return _name == other._name;
  }

public:
  static const Activation& instance(const std::string& name);

  template<typename Function>
  static bool registerFunction(const std::string& name)
  {
    return _registry.emplace(Activation(name, &Function::fn, &Function::df)).second;
  }

protected:
  typedef std::function<double(double)> function_t;

  template<typename Function>
  static Activation make(const std::string& name)
  {
    return Activation(name, &Function::fn, &Function::df);
  }

  Activation(const std::string& name, const function_t& fn, const function_t& df);

  Activation(const std::string& name): _name(name) {}

private:
  const std::string _name;
  const function_t _fn;
  const function_t _df;

private:
  static std::set<Activation> _registry;
};

} // namespace ccml

#endif