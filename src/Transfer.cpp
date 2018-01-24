#include <algorithm>
#include <cmath>
#include <sstream>

#include <iostream>

#include <unordered_map>

#include <Transfer.hpp>

namespace ccml {

const std::string& TransferFunction::name() const noexcept
{
  return _name;
}

void TransferFunction::apply(const array_t& x, array_t& y) const
{
  y.resize(x.size());
  std::transform(x.begin(), x.end(), y.begin(), [this](value_t xi) {
    return apply(xi);
  });
}

void TransferFunction::deriverate(const array_t& y, array_t& dx) const
{
  dx.resize(y.size());
  std::transform(y.begin(), y.end(), dx.begin(), [this](value_t yi) {
    return apply(yi);
  });
}

value_t TransferFunction::apply(value_t x) const
{
  return x;
}

value_t TransferFunction::deriverate(value_t y) const
{
  return 1;
}

namespace transfer {

const Transfer& heaviside()
{
  static const Transfer transfer("heaviside", 
    [](value_t x) {
      return (x > 0.0) ? 1.0 : 0.0;
    },
    [](value_t x) {
      return 0;
    },
    [](value_t y) {
      return 0;
    });
  return transfer;
}

const Transfer& identity()
{
  static const Transfer transfer("identity", 
    [](value_t x) {
      return x;
    },
    [](value_t x) {
      return 1;
    },
    [](value_t y) {
      return 1;
    });
  return transfer;
}

const Transfer& sigmoid()
{
  static auto operation = [](value_t x) {
    return 1 / (1 + std::exp(-x));
  };
  static auto derivativeFromY = [](value_t y) {
    return y * (1 - y);
  };
  static auto derivativeFromX = [=](value_t x) {
    return derivativeFromY(operation(x));
  };
  static const Transfer transfer("sigmoid", operation, derivativeFromX, derivativeFromY);
  return transfer;
}

const Transfer& relu()
{
  static auto operation = [=](value_t x) {
    return std::max(x, 0.0);
  };
  static auto derivative = [](value_t x) {
    return (x > 0.0) ? 1.0 : 0.0;
  };
  static const Transfer transfer("relu", operation, derivative, derivative);
  return transfer;
}

Transfer leakingRelu(value_t leakingRate)
{ 
  auto operation = [=](value_t x) {
    return (x > 0) ? x : (leakingRate * x);
  };
  auto derivative = [=](value_t x) {
    return (x > 0) ? 1 : leakingRate;
  };
  std::stringstream name;
  name << "leakingRelu(" << leakingRate << ")";
  return std::move(Transfer(name.str(), operation, derivative, derivative));
}

namespace {

using non_param_transfer_registry_t = std::unordered_map<std::string, Transfer>;

static non_param_transfer_registry_t nonParamTransfer = {
  {heaviside().name, heaviside()},
  {identity().name, identity()},
  {sigmoid().name, sigmoid()},
  {relu().name, relu()}
};

using param_transfer_registry_t = std::unordered_map<std::string, param_transfer_creator_t>;
static param_transfer_registry_t paramTransfer = {
  {"leakingRelu", [](const std::string& param) { return std::move(leakingRelu(std::stod(param)));} }
};

} // namespace


Transfer create(const std::string& name)
{
  const size_t parenthesisPos = name.find('(');
  if(parenthesisPos == std::string::npos)
  {
    auto transferIt = nonParamTransfer.find(name);
    if(transferIt == nonParamTransfer.end())
    {
      throw std::range_error("Transfer " + name + " not registered");
    }
    return transferIt->second;
  }
  else
  {
    auto transferIt = paramTransfer.find(name.substr(0,parenthesisPos));
    if(transferIt == paramTransfer.end())
    {
      throw std::range_error("Creator for transfer " + name + " not registered");
    }
    const size_t parenthesisEnd = name.find_last_of(')');
    if((parenthesisPos > parenthesisEnd) || (parenthesisEnd == std::string::npos))
    {
      throw std::logic_error("Invalid transfer name: " + name);
    }
    return transferIt->second(name.substr(parenthesisPos+1, parenthesisEnd - parenthesisPos - 1));
  }
}

bool registerTransfer(const Transfer& transfer)
{
  return nonParamTransfer.insert(std::make_pair(transfer.name, transfer)).second;
}

bool registerTransfer(const std::string& name, const param_transfer_creator_t& creator)
{
  return paramTransfer.insert(std::make_pair(name, creator)).second;
}

} // namespace transfer
} // namespace ccml