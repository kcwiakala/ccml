#include <layer/FullyConnectedLayer.hpp>

#include "MultiLayerPerceptron.hpp"

namespace ccml {

MultiLayerPerceptron::MultiLayerPerceptron(size_t inputSize, const transfer_ptr_t& transfer):
  _inputSize(inputSize), _transfer(transfer)
{

}

MultiLayerPerceptron& MultiLayerPerceptron::push(size_t layerSize)
{
  const size_t inputSize((size() > 0) ? outputSize() : _inputSize);
  Network::push(std::make_shared<FullyConnectedLayer>(inputSize, layerSize, _transfer));
  return *this;
}

} // namespace ccml