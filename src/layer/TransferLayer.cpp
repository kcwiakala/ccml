#include <algorithm>
#include <iostream>

#include "TransferLayer.hpp"

namespace ccml {

TransferLayer::TransferLayer(size_t layerSize, const transfer_ptr_t& transfer):
  _size(layerSize), _transfer(transfer)
{
}

TransferLayer::TransferLayer(size_t layerSize, transfer_ptr_t&& transfer):
  _size(layerSize), _transfer(std::move(transfer))
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
  _transfer->deriverate(y, e);
  std::transform(e.cbegin(), e.cend(), dy.cbegin(), e.begin(), [](value_t ei, value_t dyi) {
    return dyi * ei;
  });
}

void TransferLayer::output(const array_t& x, array_t& y) const
{
  _transfer->apply(x, y);
}

void TransferLayer::backpropagate(const array_t& error, array_t& inputError) const
{
  inputError = error;
}

void TransferLayer::toStream(std::ostream& stream) const
{
  stream << "TransferLayer:{s:" << _size << ",t:" << _transfer->name() << "}";
}

const Transfer& TransferLayer::transfer() const
{
  return *_transfer;
}

} // namespace ccml