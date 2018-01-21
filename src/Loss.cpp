#include <iostream>
#include <numeric>

#include <Network.hpp>

#include <loss/CrossEntropy.hpp>
#include <loss/MeanSquaredError.hpp>

namespace ccml {

value_t Loss::compute(const Network& network, const sample_list_t& samples) const
{
  return std::accumulate(samples.begin(), samples.end(), 0.0, [&](value_t sum, const Sample& sample){
    return sum + compute(network, sample);
  }) / samples.size();
}

loss_ptr_t Loss::meanSquaredError()
{
  static loss_ptr_t loss(std::make_shared<loss::MeanSquaredError>());
  return loss;
}

loss_ptr_t Loss::crossEntropy()
{
  static loss_ptr_t loss(std::make_shared<loss::CrossEntropy>());
  return loss;
}

} // namespace ccml