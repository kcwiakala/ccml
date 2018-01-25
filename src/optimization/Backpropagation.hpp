#ifndef CCML_BACKPROPAGATION_HPP
#define CCML_BACKPROPAGATION_HPP

#include "NetworkOptimization.hpp"

namespace ccml {

class Backpropagation: public NetworkOptimization
{
protected:
  template<typename Tl>
  Backpropagation(Network& network, Tl&& loss):
    NetworkOptimization(network, std::forward<Tl>(loss))
  {}

  void passSample(const Sample& sample, array_2d_t& activation, array_2d_t& error) const;

private:
  void feed(const Sample& sample, array_2d_t& activation, array_t& error) const;

  void backpropagate(const array_2d_t& activation, array_2d_t& error) const;
};

} // namespace ccml

#endif // CCML_BACKPROPAGATION_HPP