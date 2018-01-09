#ifndef CCML_FULLY_CONNECTED_LAYER
#define CCML_FULLY_CONNECTED_LAYER

#include "NeuronLayer.hpp"

namespace ccml {

class FullyConnectedLayer: public NeuronLayer
{
public:
  FullyConnectedLayer(size_t inputSize, size_t outputSize, const Activation& activation);

  virtual size_t inputSize() const;

  virtual void activate(const array_t& x, array_t& y);

private:
  const size_t _inputSize;
};

} // namespace ccml

#endif // CCML_FULLY_CONNECTED_LAYER