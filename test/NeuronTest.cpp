#include <gtest/gtest.h>

#include <Neuron.hpp>

class NeuronTest: public ::testing::Test 
{

};

TEST_F(NeuronTest, construction_test)
{
  ccml::Neuron n(10, ccml::Activation::sigmoid());
}