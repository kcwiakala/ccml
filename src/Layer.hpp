#ifndef CCML_LAYER_HPP
#define CCML_LAYER_HPP

#include <Neuron.hpp>

namespace ccml {

class Layer
{
protected:
  Layer(size_t size);

public:
  void init(const Initializer& weightInit, const Initializer& biasInit);

  void init(const Initializer& initializer);

  virtual void output(const array_t& x, array_t& y);

  virtual void backpropagate(const array_t& error, array_t& dx) const = 0;

protected:
  std::vector<Neuron> _neurons;
};

} // namespace ccml

#endif // CCML_LAYER_HPP