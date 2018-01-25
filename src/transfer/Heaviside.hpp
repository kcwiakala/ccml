#ifndef CCML_TRANSFER_HEAVISIDE_HPP
#define CCML_TRANSFER_HEAVISIDE_HPP

#include "Transfer.hpp"

namespace ccml {
namespace transfer {

class Heaviside: public Transfer
{
public:
  Heaviside();

protected:
  virtual value_t apply(value_t x) const override;

  virtual value_t deriverate(value_t y) const override;
};

} // namespace transfer
} // namespace ccml

#endif // CCML_TRANSFER_HEAVISIDE_HPP