#include <iostream>
#include <numeric>

#include <Network.hpp>

#include <loss/CrossEntropySigmoid.hpp>
#include <loss/CrossEntropySoftmax.hpp>
#include <loss/MeanSquaredError.hpp>

namespace ccml {

value_t Loss::compute(const Network& network, const sample_list_t& samples) const
{
  return std::accumulate(samples.cbegin(), samples.cend(), 0.0, [&](value_t sum, const Sample& sample){
    return sum + compute(network, sample);
  }) / samples.size();
}

loss_ptr_t Loss::meanSquaredError()
{
  static loss_ptr_t loss(std::make_shared<loss::MeanSquaredError>());
  return loss;
}

loss_ptr_t Loss::crossEntropySigmoid()
{
  static loss_ptr_t loss(std::make_shared<loss::CrossEntropySigmoid>());
  return loss;
}

loss_ptr_t Loss::crossEntropySoftmax()
{
  static loss_ptr_t loss(std::make_shared<loss::CrossEntropySoftmax>());
  return loss;
}

void Loss::validate(const Network& network) const
{
  if(network.size() == 0) 
  {
    throw std::logic_error("Loss function not valid for empty networks");
  }
}

} // namespace ccml