#include <cmath>

#include "SoftmaxLayer.hpp"

namespace ccml {

SoftmaxLayer::SoftmaxLayer(size_t layerSize):
  _size(layerSize)
{
}

size_t SoftmaxLayer::inputSize() const
{
  return _size;
}

size_t SoftmaxLayer::outputSize() const
{
  return _size;
}

void SoftmaxLayer::error(const array_t& y, const array_t& dy, array_t& e) const
{
  // Propagate error back, assume that dy already took softmax transfer into account
  e = dy;
}

void SoftmaxLayer::output(const array_t& x, array_t& y) const
{
  y.resize(_size, 0.0);
  value_t sum = 0.0;
  std::transform(x.begin(), x.end(), y.begin(), [&](value_t xi) {
    const value_t exi = std::exp(xi);
    sum += exi;
    return exi;
  });
  std::transform(y.begin(), y.end(), y.begin(), [=](value_t yi) {
    return yi/sum;
  }); 
}

void SoftmaxLayer::backpropagate(const array_t& error, array_t& inputError) const
{
  inputError = error;
}

void SoftmaxLayer::toStream(std::ostream& stream) const
{
  stream << "SoftmaxLayer:{s:" << _size << "}";
}

} // namespace ccml