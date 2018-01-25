#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <transfer/Sigmoid.hpp>

namespace ccml {

using ::testing::ContainerEq;
using ::testing::ElementsAre;
using ::testing::DoubleNear;

class SigmoidTest: public ::testing::Test
{
protected:
  SigmoidTest(): _sut(std::make_shared<transfer::Sigmoid>())
  {
  }

  transfer_ptr_t _sut;
};

TEST_F(SigmoidTest, name_check)
{
  EXPECT_EQ(_sut->name(), "sigmoid");
}

TEST_F(SigmoidTest, apply_check)
{
  array_t x = {1.0, 1e9, 0.0, -1e9, -1.0};
  array_t y;

  _sut->apply(x, y);
  EXPECT_THAT(y, ElementsAre(
    DoubleNear(0.7311, 1e-4), 
    DoubleNear(1.0, 1e-4), 
    DoubleNear(0.5, 1e-9), 
    DoubleNear(0.0, 1e-4), 
    DoubleNear(0.2689, 1e-4)));

  _sut->apply(x);
  EXPECT_THAT(x, ContainerEq(y));
}

TEST_F(SigmoidTest, derivativeCheck)
{
  array_t y = {0.0, 1.0, 0.5, 0.8, 0.3};
  array_t dx;
  
  _sut->deriverate(y, dx);

  EXPECT_THAT(dx, ElementsAre(
    DoubleNear(0.0, 1e-9), 
    DoubleNear(0.0, 1e-9), 
    DoubleNear(0.25, 1e-9), 
    DoubleNear(0.16, 1e-9), 
    DoubleNear(0.21, 1e-9)));
}

} // namespace ccml