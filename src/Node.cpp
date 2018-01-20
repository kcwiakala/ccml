#include <algorithm>
#include <numeric>

#include "Node.hpp"

namespace ccml {

Node::Node(size_t inputSize):
  _bias(0.0), 
  _weights(inputSize, 0.0)
{
}

void Node::init(const initializer_t& weightInit, const initializer_t& biasInit)
{
  std::generate(_weights.begin(), _weights.end(), std::cref(weightInit));
  _bias = biasInit();
}

value_t Node::output(const array_t& input) const
{
  return std::inner_product(input.begin(), input.end(), _weights.begin(), _bias);
}

void Node::adjust(const array_t& deltaWeight, double deltaBias)
{
  std::transform(deltaWeight.begin(), deltaWeight.end(), _weights.begin(), _weights.begin(), std::plus<double>());
  _bias += deltaBias;
}

void Node::toStream(std::ostream& stream) const
{
  stream << "{w:[";
  for(size_t i=0; i<_weights.size(); ++i) 
  {
    stream << ((i>0) ? "," : "") << _weights[i];
  }
  stream << "],b:" << _bias << "}";
}

} // namespace ccml