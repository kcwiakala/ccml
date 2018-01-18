#ifndef CCML_BACKPROPAGATION_HPP
#define CCML_BACKPROPAGATION_HPP

#include "NetworkOptimization.hpp"

namespace ccml {

class Backpropagation: public NetworkOptimization
{
protected:
  Backpropagation(Network& network, const loss_ptr_t& loss);

  void passSample(const Sample& sample, array_2d_t& activation, array_2d_t& error) const;

private:
  void feed(const Sample& sample, array_2d_t& activation, array_t& error) const;

  void backpropagate(const array_2d_t& activation, array_2d_t& error) const;
};

} // namespace ccml

#endif // CCML_BACKPROPAGATION_HPP