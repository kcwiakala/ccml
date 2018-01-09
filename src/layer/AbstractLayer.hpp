#ifndef CCML_ABSTRACT_LAYER_HPP
#define CCML_ABSTRACT_LAYER_HPP

#include <Types.hpp>

namespace ccml {

class AbstractLayer 
{
public:
  virtual size_t inputSize() const = 0;

  virtual size_t outputSize() const = 0;

  virtual void activate(const array_t& x, array_t& y) = 0;
};

} // namespace ccml

#endif // CCML_ABSTRACT_LAYER_HPP