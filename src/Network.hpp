#ifndef CCML_NETWORK_HPP
#define CCML_NETWORK_HPP

#include <vector>

#include <layer/AbstractLayer.hpp>

namespace ccml {

class Network 
{
public:
  template<typename Layer, typename ...Args>
  void push(Args ...args) 
  {
    push(std::make_shared<Layer>(args...));
  }

  void push(const layer_ptr_t&& layer);

  size_t inputSize() const;

  size_t outputSize() const;

  size_t size() const;

  layer_ptr_t layer(size_t idx) const;

public:
  void output(const array_t& x, array_t& y) const;

  void output(const array_t& x, array_2d_t& y) const;

  array_t output(const array_t& x) const;

private:
  std::vector<layer_ptr_t> _layers;
};

} // namespace ccml

#endif // CCML_NETWORK_HPP