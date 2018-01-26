#ifndef CCML_ABSTRACT_LOSS_HPP
#define CCML_ABSTRACT_LOSS_HPP

#include <memory>

#include <Sample.hpp>
#include <Types.hpp>

namespace ccml {

class Network;

class AbstractLoss
{
public:
  virtual ~AbstractLoss() {}

  value_t compute(const Network& network, const sample_list_t& samples) const;

  virtual value_t compute(const Network& network, const Sample& sample) const = 0;

  virtual void error(const array_t& predicted, const array_t& expected, array_t& error) const = 0;

  virtual bool includesTransfer() const;

  virtual void validate(const Network& network) const;
};

using loss_ptr_t = std::shared_ptr<AbstractLoss>;

} // namespace ccml

#endif // CCML_ABSTRACT_LOSS_HPP