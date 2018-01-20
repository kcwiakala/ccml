#include <algorithm>

#include "TransferLayer.hpp"

namespace ccml {

TransferLayer::TransferLayer(size_t layerSize, const Transfer& transfer):
  _size(layerSize), _transfer(transfer)
{
}

size_t TransferLayer::inputSize() const
{
  return _size;
}

size_t TransferLayer::outputSize() const
{
  return _size;
}

void TransferLayer::error(const array_t& y, const array_t& dy, array_t& e) const
{
  e.resize(dy.size());
  std::transform(y.begin(), y.end(), dy.begin(), e.begin(), [&](value_t yi, value_t dyi) {
    return dyi * _transfer.derivativeFromY(yi);
  });
}

void TransferLayer::output(const array_t& x, array_t& y) const
{
  y.resize(_size, 0.0);
  std::transform(x.begin(), x.end(), y.begin(), _transfer.operation);
}

void TransferLayer::backpropagate(const array_t& error, array_t& inputError) const
{
  inputError = error;
}

void TransferLayer::toStream(std::ostream& stream) const
{
  stream << "TransferLayer:{s:" << _size << ",t:" << _transfer.name << "}";
}

} // namespace ccml