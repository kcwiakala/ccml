#ifndef CCML_LOSS_CROSS_ENTROPY_SIGMOID_HPP
#define CCML_LOSS_CROSS_ENTROPY_SIGMOID_HPP

#include "AbstractLoss.hpp"

namespace ccml {
namespace loss {

class CrossEntropySigmoid: public AbstractLoss
{
public:
  virtual value_t compute(const Network& network, const Sample& sample) const override;

  virtual void error(const array_t& predicted, const array_t& expected, array_t& error) const override;

  virtual void validate(const Network& network) const override;

  virtual bool includesTransfer() const override;
};

} // namespace loss
} // namespace ccml

#endif // CCML_LOSS_CROSS_ENTROPY_SIGMOID_HPP