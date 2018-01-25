#ifndef CCML_TRANSFER_SIGMOID_HPP
#define CCML_TRANSFER_SIGMOID_HPP

#include "Transfer.hpp"

namespace ccml {
namespace transfer {

class Sigmoid: public Transfer
{
public:
  Sigmoid();
  
protected:
  virtual value_t apply(value_t x) const override;

  virtual value_t deriverate(value_t y) const override;
};

} // namespace transfer
} // namespace ccml

#endif // CCML_TRANSFER_SIGMOID_HPP