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

public:
  void output(const array_t& x, array_t& y);

private:
  std::vector<layer_ptr_t> _layers;
};

} // namespace ccml

#endif // CCML_NETWORK_HPP