#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <transfer/Relu.hpp>

namespace ccml {

using ::testing::ContainerEq;
using ::testing::ElementsAre;
using ::testing::DoubleNear;

class ReluTest: public ::testing::Test
{
protected:
  ReluTest(): _sut(std::make_shared<transfer::Relu>())
  {
  }

  transfer_ptr_t _sut;
};

TEST_F(ReluTest, name_check)
{
  EXPECT_EQ(_sut->name(), "relu");
}

TEST_F(ReluTest, apply_check)
{
  array_t x = {1.0, 1e-6, 0.0, -4.21, 17.321};
  array_t y;

  _sut->apply(x, y);
  EXPECT_THAT(y, ElementsAre(
    DoubleNear(1.0, 1e-9), 
    DoubleNear(1e-6, 1e-9), 
    DoubleNear(0.0, 1e-9), 
    DoubleNear(0.0, 1e-9), 
    DoubleNear(17.321, 1e-9)));

  _sut->apply(x);
  EXPECT_THAT(x, ContainerEq(y));
}

TEST_F(ReluTest, derivativeCheck)
{
  array_t y = {10.0, 1.0, 0.0, -18.0, 7.3};
  array_t dx;
  
  _sut->deriverate(y, dx);

  EXPECT_THAT(dx, ElementsAre(
    DoubleNear(1.0, 1e-9), 
    DoubleNear(1.0, 1e-9), 
    DoubleNear(0.0, 1e-9), 
    DoubleNear(0.0, 1e-9), 
    DoubleNear(1.0, 1e-9)));
}

} // namespace ccml