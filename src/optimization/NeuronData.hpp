#ifndef CCML_NEURON_DATA_HPP
#define CCML_NEURON_DATA_HPP

#include <memory>

namespace ccml {

class NeuronData
{
public:
  virtual ~NeuronData() {};

  virtual void reset() = 0;
};

typedef std::unique_ptr<NeuronData> neuron_data_ptr_t;

} // namespace ccml

#endif // CCML_NEURON_DATA_HPP