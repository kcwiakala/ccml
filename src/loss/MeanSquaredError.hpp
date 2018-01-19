#ifndef CCML_LOSS_MEAN_SQUARED_ERROR_HPP
#define CCML_LOSS_MEAN_SQUARED_ERROR_HPP

#include <Loss.hpp>

namespace ccml {
namespace loss {

class MeanSquaredError: public Loss
{
public:
  virtual value_t compute(const Network& network, const Sample& sample) const;

  virtual value_t error(value_t predicted, value_t expected) const;
};

} // namespace loss
} // namespace ccml

#endif // CCML_LOSS_MEAN_SQUARED_ERROR_HPP