#ifndef CCML_TYPED_LAYER_HPP
#define CCML_TYPED_LAYER_HPP

#include "AbstractLayer.hpp"

namespace ccml {

class TypedLayer: public AbstractLayer
{
protected:
  TypedLayer(const std::string& type): _type(type) {}

  virtual const std::string& type() const
  {
    return _type;
  }
  
private:
  std::string _type;
};

} // namespace ccml

#endif // CCML_TYPED_LAYER_HPP