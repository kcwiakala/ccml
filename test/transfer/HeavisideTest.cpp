#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <transfer/Heaviside.hpp>

namespace ccml {

using ::testing::ContainerEq;
using ::testing::ElementsAre;
using ::testing::DoubleNear;

class HeavisideTest: public ::testing::Test
{
protected:
  HeavisideTest(): _sut(std::make_shared<transfer::Heaviside>())
  {
  }

  transfer_ptr_t _sut;
};

TEST_F(HeavisideTest, name_check)
{
  EXPECT_EQ(_sut->name(), "heaviside");
}

TEST_F(HeavisideTest, apply_check)
{
  array_t x = {1.0, 1e-6, 0.0, -1e-6, -1.0};
  array_t y;

  _sut->apply(x, y);
  EXPECT_THAT(y, ElementsAre(
    DoubleNear(1.0, 1e-9), 
    DoubleNear(1.0, 1e-9), 
    DoubleNear(0.0, 1e-9), 
    DoubleNear(0.0, 1e-9), 
    DoubleNear(0.0, 1e-9)));

  _sut->apply(x);
  EXPECT_THAT(x, ContainerEq(y));
}

TEST_F(HeavisideTest, derivativeCheck)
{
  array_t y = {10.0, 1.0, -0.5, -18.0, 7.3};
  array_t dx;
  
  _sut->deriverate(y, dx);

  EXPECT_THAT(dx, ElementsAre(
    DoubleNear(0.0, 1e-9), 
    DoubleNear(0.0, 1e-9), 
    DoubleNear(0.0, 1e-9), 
    DoubleNear(0.0, 1e-9), 
    DoubleNear(0.0, 1e-9)));
}

} // namespace ccml