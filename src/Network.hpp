#ifndef CCML_NETWORK_HPP
#define CCML_NETWORK_HPP

#include <functional>
#include <vector>
#include <layer/NeuronLayer.hpp>

namespace ccml {

class Network 
{
public:
  Network() {}

  template<typename Layer, typename... Args>
  void push(Args&&... args) 
  {
    push(std::make_shared<Layer>(std::forward<Args>(args)...));
  }

  void push(const layer_ptr_t& layer);

  size_t inputSize() const;

  size_t outputSize() const;

public:
  void output(const array_t& x, array_t& y) const;

  void output(const array_t& x, array_2d_t& y) const;

  array_t output(const array_t& x) const;

public:
  typedef std::pair<layer_ptr_t, bool> layer_type_pair_t;
  typedef std::vector<layer_type_pair_t> layer_list_t;
  
  size_t size() const;

  layer_ptr_t outputLayer() const;

  layer_ptr_t layer(size_t idx) const;

  neuron_layer_ptr_t neuronLayer(size_t idx) const;

private:
  layer_list_t _layers;
};

} // namespace ccml

#endif // CCML_NETWORK_HPP