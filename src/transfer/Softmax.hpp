#ifndef CCML_TRANSFER_SOFTMAX_HPP
#define CCML_TRANSFER_SOFTMAX_HPP

#include "Transfer.hpp"

namespace ccml {
namespace transfer {

class Softmax: public Transfer
{
public:
  Softmax();

protected:
  virtual void apply(array_t& x) const override;

  virtual void apply(const array_t& x, array_t& y) const override;

  virtual void deriverate(const array_t& y, array_t& dx) const override;
};

} // namespace transfer
} // namespace ccml

#endif // CCML_TRANSFER_SOFTMAX_HPP