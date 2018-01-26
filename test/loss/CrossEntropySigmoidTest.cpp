#include <CcmlTest.hpp>

#include <loss/CrossEntropySigmoid.hpp>
#include <layer/FullyConnectedLayer.hpp>
#include <layer/TransferLayer.hpp>
#include <Network.hpp>
#include <transfer/Relu.hpp>
#include <transfer/Sigmoid.hpp>

namespace ccml {

using ::testing::EndsWith;

class CrossEntropySigmoidTest: public CcmlTest
{
protected:
  CrossEntropySigmoidTest():
    _sut(new loss::CrossEntropySigmoid())
  {
  }

  loss_ptr_t _sut;
};

TEST_F(CrossEntropySigmoidTest, validation)
{
  Network network;
  EXPECT_THROW_MATCHING(_sut->validate(network), std::logic_error, EndsWith("not valid for empty networks"));

  network.push<TransferLayer>(2, std::make_shared<transfer::Relu>());
  EXPECT_THROW_MATCHING(_sut->validate(network), std::logic_error, EndsWith("compatible only with networks having Sigmoid output"));

  network.push<TransferLayer>(2, std::make_shared<transfer::Sigmoid>());
  EXPECT_THROW_MATCHING(_sut->validate(network), std::logic_error, EndsWith("compatible only with networks having single output"));

  network.push<FullyConnectedLayer>(2, 1, std::make_shared<transfer::Sigmoid>());
  EXPECT_NO_THROW(_sut->validate(network));
}

} // namespace ccml