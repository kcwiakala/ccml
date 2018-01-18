#ifndef CCML_MULTI_LAYER_PERCEPTRON 
#define CCML_MULTI_LAYER_PERCEPTRON

#include <Network.hpp>

namespace ccml {

class MultiLayerPerceptron: public Network
{
public:
  MultiLayerPerceptron(size_t inputSize, const Activation& activation);

  MultiLayerPerceptron& push(size_t layerSize);

private:
  using Network::push;

private:
  const size_t _inputSize;
  const Activation& _activation;
};

} // namespace ccml

#endif // CCML_MULTI_LAYER_PERCEPTRON