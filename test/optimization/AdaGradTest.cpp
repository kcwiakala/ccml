#include <memory>

#include <optimization/BaseOptimizationTest.hpp>

#include <Network.hpp>
#include <loss/MeanSquaredError.hpp>
#include <transfer/Sigmoid.hpp>
#include <layer/FullyConnectedLayer.hpp>

#include <optimization/AdaGrad.hpp>

namespace ccml {

class AdaGradTest: public BaseOptimizationTest<AdaGrad>
{
protected:

};

TEST_F(AdaGradTest, learn_sin_function)
{
  auto l1 = std::make_shared<FullyConnectedLayer>(1, 3, std::make_shared<transfer::Sigmoid>());
  auto l2 = std::make_shared<FullyConnectedLayer>(3, 1, std::make_shared<transfer::Sigmoid>());
  l1->init(initializer::uniform(0.5, 1), initializer::constant(0));
  l2->init(initializer::uniform(0.5, 1), initializer::constant(0));

  Network net;
  net.push(l1);
  net.push(l2);

  _sut = std::make_unique<AdaGrad>(net, std::make_shared<loss::MeanSquaredError>(), 0.01);

  EXPECT_TRUE(_sut->train(sinus(100), 10, 10000, 1e-3));
  validate(net, sinus(10), 0.2);
}

} // namespace ccml