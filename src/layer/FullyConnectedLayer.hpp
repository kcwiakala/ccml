#ifndef CCML_FULLY_CONNECTED_LAYER
#define CCML_FULLY_CONNECTED_LAYER

#include "NeuronLayer.hpp"

namespace ccml {

class FullyConnectedLayer: public NeuronLayer
{
public:
  template<typename Tr>
  FullyConnectedLayer(size_t inputSize, size_t outputSize, Tr&& transfer):
    NeuronLayer("FullyConnectedLayer", outputSize, inputSize, std::forward<Tr>(transfer)),
    _inputSize(inputSize)
  {}

  virtual size_t inputSize() const;

  virtual void output(const array_t& x, array_t& y) const;

  virtual void backpropagate(const array_t& error, array_t& inputError) const;

  virtual void splitError(const array_t& x, const array_t& error, const error_reader_t& reader) const;

private:
  const size_t _inputSize;
};

} // namespace ccml

#endif // CCML_FULLY_CONNECTED_LAYER