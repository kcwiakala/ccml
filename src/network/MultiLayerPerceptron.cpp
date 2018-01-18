#include <layer/FullyConnectedLayer.hpp>

#include "MultiLayerPerceptron.hpp"

namespace ccml {

MultiLayerPerceptron::MultiLayerPerceptron(size_t inputSize, const Activation& activation):
  _inputSize(inputSize), _activation(activation)
{

}

MultiLayerPerceptron& MultiLayerPerceptron::push(size_t layerSize)
{
  const size_t inputSize((size() > 0) ? outputSize() : _inputSize);
  Network::push(std::make_shared<FullyConnectedLayer>(inputSize, layerSize, _activation));
  return *this;
}

} // namespace ccml