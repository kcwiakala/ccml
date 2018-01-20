#ifndef CCML_TRANSFER_HPP
#define CCML_TRANSFER_HPP

#include <memory>

#include <Serializable.hpp>
#include <Types.hpp>

namespace ccml {

struct Transfer
{
public:
  Transfer(const std::string& nm, const value_converter_t& op, const value_converter_t& dfx, const value_converter_t& dfy):
    name(nm), operation(op), derivativeFromX(dfx), derivativeFromY(dfy)
  {}

  const std::string name;
  const value_converter_t operation;
  const value_converter_t derivativeFromX;
  const value_converter_t derivativeFromY;
};

namespace transfer {

const Transfer& identity();

const Transfer& sigmoid();

const Transfer& heaviside();

const Transfer& relu();

Transfer leakingRelu(value_t leakingRate = 0.01);

} // namespace transfer
} // namespace ccml

#endif // CCML_TRANSFER_HPP