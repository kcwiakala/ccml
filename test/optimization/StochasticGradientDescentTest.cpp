#include <iostream>
#include <random>

#include <gtest/gtest.h>

#include <layer/FullyConnectedLayer.hpp>
#include <layer/TransferLayer.hpp>
#include <loss/CrossEntropySigmoid.hpp>
#include <optimization/StochasticGradientDescent.hpp>
#include <optimization/Momentum.hpp>
#include <optimization/Adam.hpp>
#include <transfer/LeakingRelu.hpp>
#include <transfer/Relu.hpp>
#include <transfer/Sigmoid.hpp>

namespace ccml {

class StochasticGradientDescentTest: public testing::Test
{
protected:
  std::unique_ptr<StochasticGradientDescent> _optimizer;
};

TEST_F(StochasticGradientDescentTest, simple)
{ 
  Network net;
  neuron_layer_ptr_t l1 = std::make_shared<FullyConnectedLayer>(2, 3, std::make_shared<transfer::LeakingRelu>(0.01));
  neuron_layer_ptr_t l2 = std::make_shared<FullyConnectedLayer>(3, 1, std::make_shared<transfer::Sigmoid>());
  // layer_ptr_t l3 = std::make_shared<TransferLayer>(1, transfer::sigmoid());
  net.push(l1);
  net.push(l2);
  // net.push(l3);

  l1->init(initializer::uniform(0.5, 1), initializer::constant(0));
  l2->init(initializer::uniform(0.1, 0.3), initializer::constant(0));

  // l3->init(initializer, initializer);

  // std::cout << l1 << std::endl;
  // std::cout << l2 << std::endl;

  // l3->init(Initializer::uniform(-0.2, -0.5), Initializer::uniform(-0.2, -0.5));
  loss_ptr_t loss = std::make_shared<loss::CrossEntropySigmoid>();
  // _optimizer = std::make_unique<StochasticGradientDescent>(net, loss, 0.01);
  // _optimizer = std::make_unique<Momentum>(net, loss, 0.2, 0.1);
  _optimizer = std::make_unique<Adam>(net, loss, 0.01);

  array_t aux;

  sample_list_t xorSamples = {
    {{1,0}, {1}},
    {{0,1}, {1}},
    {{1,1}, {0}},
    {{0,0}, {0}}
  };

  // std::random_device rd;
  // std::mt19937 g(rd());

  // std::shuffle(xorSamples.begin(), xorSamples.end(), g);
  
  // std::cout << "Loss before training: " << loss->compute(net, xorSamples) << std::endl;
  bool success = false;
  // for(size_t i=0; i<100 && !success; ++i) 
  // {
  //   // std::cout << "loop" << std::endl;
  //   l1->init(initializer, initializer);
  //   l2->init(initializer, initializer);
  //   // l3->init(initializer, initializer);
  //   success = _optimizer->train(xorSamples, 100, 0.05);
  // };
  success = _optimizer->train(xorSamples, 4, 10000, 0.001);
  EXPECT_TRUE(success);

  // std::cout << l1 << std::endl;
  // std::cout << l2 << std::endl;
  // std::cout << "Loss after training: " << loss->compute(net, xorSamples) << std::endl;

  EXPECT_GT(net.output(xorSamples[0].input)[0], 0.5);
  EXPECT_GT(net.output({0, 1})[0], 0.5);
  EXPECT_LT(net.output({1, 1})[0], 0.5);
  EXPECT_LT(net.output({0, 0})[0], 0.5);
}

} // namespace ccml