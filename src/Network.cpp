#include "Network.hpp"

namespace ccml {

void Network::push(const layer_ptr_t&& layer)
{
  if(!_layers.empty())
  {
    if(_layers.back()->outputSize() != layer->inputSize()) 
    {
      throw std::logic_error("Not matching layer input size");
    }
  }
  _layers.emplace_back(layer);
}

void Network::output(const array_t& x, array_t& y) const
{
  thread_local static array_t aux;

  if(!_layers.empty()) 
  {
    _layers[0]->output(x, y);
    for(size_t i=1; i<_layers.size();++i) {
      aux.swap(y);
      _layers[i]->output(aux, y);
    }
  }
  else
  {
    y.assign(x.begin(), x.end());
  }
}

array_t Network::output(const array_t& x) const
{
  array_t y;
  output(x,y);
  return std::move(y);
}

} // namespace ccml