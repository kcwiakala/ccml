#ifndef CCML_ABSTRACT_LAYER_HPP
#define CCML_ABSTRACT_LAYER_HPP

#include <memory>

#include <Serializable.hpp>
#include <Types.hpp>

namespace ccml {

class AbstractLayer: public Serializable 
{
public:
  virtual ~AbstractLayer() {}

  virtual size_t inputSize() const = 0;

  virtual size_t outputSize() const = 0;

  virtual void output(const array_t& x, array_t& y) const = 0;

  virtual void error(const array_t& y, const array_t& dy, array_t& e) const = 0;

  virtual void backpropagate(const array_t& error, array_t& inputError) const = 0;
};

typedef std::shared_ptr<AbstractLayer> layer_ptr_t;

} // namespace ccml

#endif // CCML_ABSTRACT_LAYER_HPP