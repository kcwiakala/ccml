#ifndef CCML_INPUT_LAYER_HPP
#define CCML_INPUT_LAYER_HPP

#include "AbstractLayer.hpp"

namespace ccml {

class InputLayer: public AbstractLayer
{
public:
  InputLayer(size_t size);

private:
  const size_t _size;
};

} // namespace ccml

#endif //  CCML_INPUT_LAYER_HPP
