#ifndef CCML_LOSS_MEAN_SQUARED_ERROR_HPP
#define CCML_LOSS_MEAN_SQUARED_ERROR_HPP

#include "AbstractLoss.hpp"

namespace ccml {
namespace loss {

class MeanSquaredError: public AbstractLoss
{
public:
  virtual value_t compute(const Network& network, const Sample& sample) const override;

  virtual void error(const array_t& predicted, const array_t& expected, array_t& error) const override;
};

} // namespace loss
} // namespace ccml

#endif // CCML_LOSS_MEAN_SQUARED_ERROR_HPP