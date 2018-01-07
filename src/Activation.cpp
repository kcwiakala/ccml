#include <map>

#include <activation/Heaviside.hpp>
#include <activation/Relu.hpp>
#include <activation/Sigmoid.hpp>

#include <Activation.hpp>

namespace ccml {

std::set<Activation> Activation::_registry = {
  Activation::make<activation::Heaviside>("heaviside"),
  Activation::make<activation::Relu>("relu"),
  Activation::make<activation::Sigmoid>("sigmoid")
};

Activation::Activation(const std::string& name, const function_t& fn, const function_t& df):
  _name(name), _fn(fn), _df(df)
{
}

const Activation& Activation::instance(const std::string& name)
{
  auto actIt = _registry.find(Activation(name));
  if(actIt == _registry.end()) 
  {
    throw std::runtime_error("Unknown activation function name: " + name);
  }
  return *actIt;
}

const Activation& Activation::heaviside()
{
  static const Activation& act = instance("heaviside");
  return act;
}

const Activation& Activation::relu()
{
  static const Activation& act = instance("relu");
  return act;
}

const Activation& Activation::sigmoid()
{
  static const Activation& act = instance("sigmoid");
  return act;
}

} // namespace ccml