#ifndef CCML_LOSS_HPP
#define CCML_LOSS_HPP

#include <memory>

#include <Sample.hpp>
#include <Types.hpp>

namespace ccml {

class Network;
class Loss;

typedef std::shared_ptr<Loss> loss_ptr_t;

class Loss
{
public:
  value_t compute(const Network& network, const sample_list_t& samples) const;

  virtual value_t compute(const Network& network, const Sample& sample) const = 0;

  virtual value_t error(value_t predicted, value_t expected) const = 0;

  virtual void validate(const Network& network) const;

public:
  static loss_ptr_t meanSquaredError();

  static loss_ptr_t crossEntropySigmoid();

  static loss_ptr_t crossEntropySoftmax();
};

} // namespace ccml

#endif // CCML_LOSS_HPP