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
  _neuronLayers.push_back(static_cast<bool>(std::dynamic_pointer_cast<NeuronLayer>(layer)));
  _layers.emplace_back(layer);
}

size_t Network::inputSize() const
{
  return _layers.empty() ? 0 : _layers.front()->inputSize();
}

size_t Network::outputSize() const
{
  return _layers.empty() ? 0 : _layers.back()->outputSize();
}

size_t Network::size() const
{
  return _layers.size();
}

layer_ptr_t Network::layer(size_t idx) const
{
  return _layers.at(idx);
}

neuron_layer_ptr_t Network::neuronLayer(size_t idx) const
{
  return (_neuronLayers.at(idx)) ? std::static_pointer_cast<NeuronLayer>(_layers.at(idx)) : 0;
}

void Network::output(const array_t& x, array_t& y) const
{
  thread_local static array_t aux;

  if(!_layers.empty()) 
  {
    _layers[0]->output(x, y);
    for(size_t i=1; i<_layers.size();++i) 
    {
      aux.swap(y);
      _layers[i]->output(aux, y);
    }
  }
  else
  {
    y.assign(x.begin(), x.end());
  }
}

void Network::output(const array_t& x, array_2d_t& y) const
{
  if(!_layers.empty()) 
  {
    y.resize(_layers.size());
    _layers[0]->output(x, y[0]);
    for(size_t i=1; i<_layers.size();++i) 
    {
      _layers[i]->output(y[i-1], y[i]);
    }
  }
  else
  {
    y.assign(1, x);
  }
}

array_t Network::output(const array_t& x) const
{
  array_t y;
  output(x,y);
  return std::move(y);
}

} // namespace ccml