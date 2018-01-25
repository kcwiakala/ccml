#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <transfer/Identity.hpp>

namespace ccml {

using ::testing::ContainerEq;
using ::testing::ElementsAre;
using ::testing::DoubleNear;

class IdentityTest: public ::testing::Test
{
protected:
  IdentityTest(): _sut(std::make_shared<transfer::Identity>())
  {
  }

  transfer_ptr_t _sut;
};

TEST_F(IdentityTest, name_check)
{
  EXPECT_EQ(_sut->name(), "identity");
}

TEST_F(IdentityTest, apply_check)
{
  array_t x = {1.0, 1e-6, 0.0, -7.23, 9.2};
  array_t y;

  _sut->apply(x, y);
  EXPECT_THAT(y, ElementsAre(
    DoubleNear(1.0, 1e-9), 
    DoubleNear(1e-6, 1e-9), 
    DoubleNear(0.0, 1e-9), 
    DoubleNear(-7.23, 1e-9), 
    DoubleNear(9.2, 1e-9)));

  _sut->apply(x);
  EXPECT_THAT(x, ContainerEq(y));
}

TEST_F(IdentityTest, derivativeCheck)
{
  array_t y = {10.0, 1.0, -0.5, -18.0, 7.3};
  array_t dx;
  
  _sut->deriverate(y, dx);

  EXPECT_THAT(dx, ElementsAre(
    DoubleNear(1.0, 1e-9), 
    DoubleNear(1.0, 1e-9), 
    DoubleNear(1.0, 1e-9), 
    DoubleNear(1.0, 1e-9), 
    DoubleNear(1.0, 1e-9)));
}

} // namespace ccml