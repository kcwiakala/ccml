#include <iostream>
#include <numeric>

#include <Network.hpp>

#include "AbstractLoss.hpp"

namespace ccml {

value_t AbstractLoss::compute(const Network& network, const sample_list_t& samples) const
{
  return std::accumulate(samples.cbegin(), samples.cend(), 0.0, [&](value_t sum, const Sample& sample){
    return sum + compute(network, sample);
  }) / samples.size();
}

bool AbstractLoss::includesTransfer() const
{
  return false;
}

void AbstractLoss::validate(const Network& network) const
{
  if(network.size() == 0) 
  {
    throw std::logic_error("AbstractLoss not valid for empty networks");
  }
}

} // namespace ccml