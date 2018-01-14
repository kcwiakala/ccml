#include <iostream>

#include <gtest/gtest.h>

#include <activation/Sigmoid.hpp>
#include <Neuron.hpp>

namespace ccml {
  
class NeuronTest: public ::testing::Test 
{
};

TEST_F(NeuronTest, construction_test)
{
  ccml::Neuron n(2, ccml::Activation::sigmoid());

  std::cout << n << std::endl;

  n.init(Initializer::uniform(-3,1.2), Initializer::normal(0,1));
  std::cout << n << std::endl;
}

} // namespace ccml