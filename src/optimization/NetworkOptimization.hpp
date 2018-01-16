#ifndef CCML_NETWORK_OPTIMIZATION_HPP
#define CCML_NETWORK_OPTIMIZATION_HPP

#include <Loss.hpp>
#include <Network.hpp>

namespace ccml {

class NetworkOptimization 
{
protected:
  NetworkOptimization(Network& network, const loss_ptr_t& loss): _network(network), _loss(loss) {}

  virtual ~NetworkOptimization() {}

protected:
  Network& _network;
  loss_ptr_t _loss;
};

} // namespace ccml

#endif // CCML_NETWORK_OPTIMIZATION_HPP