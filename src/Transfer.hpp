#ifndef CCML_TRANSFER_HPP
#define CCML_TRANSFER_HPP

#include <memory>
#include <ostream>

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


using param_transfer_creator_t = std::function<Transfer(const std::string&)>;

Transfer create(const std::string& name);

bool registerTransfer(const Transfer& transfer);

bool registerTransfer(const std::string& name, const param_transfer_creator_t& creator);

} // namespace transfer
} // namespace ccml

inline std::ostream& operator<<(std::ostream& stream, const ccml::Transfer& transfer)
{
  stream << transfer.name;
  return stream;
}

#endif // CCML_TRANSFER_HPP