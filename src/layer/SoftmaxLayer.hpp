#ifndef CCML_SOFTMAX_LAYER_HPP
#define CCML_SOFTMAX_LAYER_HPP

#include "AbstractLayer.hpp"

namespace ccml {
  
class SoftmaxLayer: public AbstractLayer
{
public:
  SoftmaxLayer(size_t layerSize);

  virtual size_t inputSize() const override;

  virtual size_t outputSize() const override;

  virtual void output(const array_t& x, array_t& y) const override;

  virtual void error(const array_t& y, const array_t& dy, array_t& e) const override;

  virtual void backpropagate(const array_t& error, array_t& inputError) const override;

  virtual void toStream(std::ostream& stream) const override;

private:
  const size_t _size;
};

} // namespace ccml

#endif // CCML_SOFTMAX_LAYER_HPP