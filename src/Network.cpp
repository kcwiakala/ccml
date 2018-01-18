#include "Network.hpp"

namespace ccml {

using namespace std::placeholders;

void Network::push(const layer_ptr_t&& layer)
{
  if(!_layers.empty())
  {
    if(outputSize() != layer->inputSize()) 
    {
      throw std::logic_error("Not matching layer input size");
    }
  }
  const bool isNeuronLayer(std::dynamic_pointer_cast<NeuronLayer>(layer));
  _layers.emplace_back(std::make_pair(layer, isNeuronLayer));
}

size_t Network::inputSize() const
{
  return _layers.empty() ? 0 : _layers.front().first->inputSize();
}

size_t Network::outputSize() const
{
  return _layers.empty() ? 0 : _layers.back().first->outputSize();
}

size_t Network::size() const
{
  return _layers.size();
}

layer_ptr_t Network::layer(size_t idx) const
{
  return _layers.at(idx).first;
}

neuron_layer_ptr_t Network::neuronLayer(size_t idx) const
{
  const layer_type_pair_t& layer = _layers.at(idx);
  return layer.second ? std::static_pointer_cast<NeuronLayer>(layer.first) : 0;
}

void Network::output(const array_t& x, array_t& y) const
{
  thread_local static array_t aux;

  if(_layers.empty()) 
  {
    y.clear();
  }
  else
  {
    _layers[0].first->output(x, y);
    for(size_t i=1; i<_layers.size();++i) 
    {
      aux.swap(y);
      _layers[i].first->output(aux, y);
    }
  }
}

void Network::output(const array_t& x, array_2d_t& y) const
{
  if(_layers.empty())
  {
    y.clear();
  }
  else
  {
    y.resize(_layers.size());
    _layers[0].first->output(x, y[0]);
    for(size_t i=1; i<_layers.size();++i) 
    {
      _layers[i].first->output(y[i-1], y[i]);
    }
  }
}

array_t Network::output(const array_t& x) const
{
  array_t y;
  output(x,y);
  return std::move(y);
}

} // namespace ccml