#include <iostream>

#include <gtest/gtest.h>

#include <layer/FullyConnectedLayer.hpp>
#include <optimization/StochasticGradientDescent.hpp>
#include <optimization/Momentum.hpp>

namespace ccml {

class StochasticGradientDescentTest: public testing::Test
{
protected:
  std::unique_ptr<StochasticGradientDescent> _optimizer;
};

TEST_F(StochasticGradientDescentTest, simple)
{ 
  Network net;
  neuron_layer_ptr_t l1 = std::make_shared<FullyConnectedLayer>(2, 3, Activation::sigmoid());
  neuron_layer_ptr_t l2 = std::make_shared<FullyConnectedLayer>(3, 1, Activation::sigmoid());
  // neuron_layer_ptr_t l3 = std::make_shared<FullyConnectedLayer>(2, 1, Activation::sigmoid());
  net.push(l1);
  net.push(l2);
  // net.push(l3);

  l1->init(Initializer::uniform(-0.3, -0.5), Initializer::uniform(-0.3, -0.5));
  l2->init(Initializer::uniform(-0.3, -0.5), Initializer::uniform(-0.3, -0.5));
  // l3->init(Initializer::uniform(-0.2, -0.5), Initializer::uniform(-0.2, -0.5));

  //_optimizer = std::make_unique<StochasticGradientDescent>(net, Loss::quadratic(), 0.5);
  _optimizer = std::make_unique<Momentum>(net, Loss::quadratic(), 0.5, 0.8);

  array_t aux;

  loss_ptr_t loss = Loss::quadratic();

  sample_list_t xorSamples = {
    {{1,0}, {1}},
    {{0,1}, {1}},
    {{1,1}, {0}},
    {{0,0}, {0}}
  };
  
  // std::cout << "Loss before training: " << loss->compute(net, xorSamples) << std::endl;
  const bool success = _optimizer->train(xorSamples, 100000, 0.001);
  EXPECT_TRUE(success);

  // std::cout << "Loss after training: " << loss->compute(net, xorSamples) << std::endl;

  EXPECT_GT(net.output(xorSamples[0].input)[0], 0.5);
  EXPECT_GT(net.output({0, 1})[0], 0.5);
  EXPECT_LT(net.output({1, 1})[0], 0.5);
  EXPECT_LT(net.output({0, 0})[0], 0.5);
}

} // namespace ccml