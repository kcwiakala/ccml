#ifndef CCML_TRANSFER_LAYER_HPP
#define CCML_TRANSFER_LAYER_HPP

#include <layer/AbstractLayer.hpp>
#include <transfer/Transfer.hpp>

namespace ccml {

class TransferLayer: public AbstractLayer
{
public:
  TransferLayer(size_t layerSize, const transfer_ptr_t& transfer);

  TransferLayer(size_t layerSize, transfer_ptr_t&& transfer);

  virtual size_t inputSize() const;

  virtual size_t outputSize() const;

  virtual void error(const array_t& y, const array_t& dy, array_t& e) const;

  virtual void output(const array_t& x, array_t& y) const;

  virtual void backpropagate(const array_t& error, array_t& inputError) const;

  virtual void toStream(std::ostream& stream) const;

  const Transfer& transfer() const;

private:
  const size_t _size;
  const transfer_ptr_t _transfer;
};

} // namespace ccml

#endif // CCML_TRANSFER_LAYER_HPP