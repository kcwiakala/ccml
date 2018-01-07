#include <gtest/gtest.h>

#include <activation/Sigmoid.hpp>
#include <Neuron.hpp>

class NeuronTest: public ::testing::Test 
{

};

TEST_F(NeuronTest, construction_test)
{
  ccml::Neuron n(10, ccml::activation::Sigmoid());
}