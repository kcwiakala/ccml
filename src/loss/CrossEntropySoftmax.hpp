#ifndef CCML_LOSS_CROSS_ENTROPY_SOFTMAX_HPP
#define CCML_LOSS_CROSS_ENTROPY_SOFTMAX_HPP

#include "AbstractLoss.hpp"

namespace ccml {
namespace loss {

class CrossEntropySoftmax: public AbstractLoss
{
public:
  virtual value_t compute(const Network& network, const Sample& sample) const;

  virtual value_t error(value_t predicted, value_t expected) const;

  virtual void validate(const Network& network) const override;
};

} // namespace loss
} // namespace ccml

#endif // CCML_LOSS_CROSS_ENTROPY_SOFTMAX_HPP