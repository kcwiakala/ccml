#ifndef CCML_LAYER_HPP
#define CCML_LAYER_HPP

#include <Neuron.hpp>

namespace ccml {

class Layer
{
public:
  Layer(size_t size);

public:
  void init(const Initializer& weightInit, const Initializer& biasInit);

  void init(const Initializer& initializer);

  std::vector<double> output(const std::vector<double>& x);

protected:
  std::vector<Neuron> _neurons;
};

} // namespace ccml

#endif // CCML_LAYER_HPP