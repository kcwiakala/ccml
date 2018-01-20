#ifndef CCML_MULTI_LAYER_PERCEPTRON 
#define CCML_MULTI_LAYER_PERCEPTRON

#include <Network.hpp>

namespace ccml {

class MultiLayerPerceptron: public Network
{
public:
  MultiLayerPerceptron(size_t inputSize, const Transfer& transfer);

  MultiLayerPerceptron& push(size_t layerSize);

private:
  using Network::push;

private:
  const size_t _inputSize;
  const Transfer& _transfer;
};

} // namespace ccml

#endif // CCML_MULTI_LAYER_PERCEPTRON