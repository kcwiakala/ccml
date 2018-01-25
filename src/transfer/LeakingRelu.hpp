#ifndef CCML_TRANSFER_LEAKING_RELU_HPP
#define CCML_TRANSFER_LEAKING_RELU_HPP

#include "Transfer.hpp"

namespace ccml {
namespace transfer {

class LeakingRelu: public Transfer
{
public:
  LeakingRelu(double leakingRate);

protected:
  virtual value_t apply(value_t x) const override;

  virtual value_t deriverate(value_t y) const override;

private:
  const double _leakingRate;
};

} // namespace transfer
} // namespace ccml

#endif // CCML_TRANSFER_LEAKING_RELU_HPP