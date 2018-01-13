#ifndef CCML_STOCHASTIC_GRADIENT_DESCENT_HPP
#define CCML_STOCHASTIC_GRADIENT_DESCENT_HPP

#include <Network.hpp>
#include <Sample.hpp>

namespace ccml {

class Loss
{

};

class StochasticGradientDescent 
{
public:
  StochasticGradientDescent(const Loss& loss);

  void learnSample(Network& network, const Sample& sample);

protected:
  void updateLayer(Network& network, size_t layerIdx, const array_2d_t& activation, array_t& error);

private:
  Loss _loss;
};

} // namespace ccml

#endif // CCML_STOCHASTIC_GRADIENT_DESCENT_HPP