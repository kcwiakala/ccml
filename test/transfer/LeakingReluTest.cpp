#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <transfer/LeakingRelu.hpp>

namespace ccml {

using ::testing::ContainerEq;
using ::testing::ElementsAre;
using ::testing::DoubleNear;
using ::testing::MatchesRegex;

class LeakingReluTest: public ::testing::Test
{
protected:
  LeakingReluTest(): 
    _leakingRate(0.01),
    _sut(std::make_shared<transfer::LeakingRelu>(_leakingRate))
  {
  }

  const double _leakingRate;
  transfer_ptr_t _sut;
};

TEST_F(LeakingReluTest, name_check)
{
  EXPECT_THAT(_sut->name(), MatchesRegex("leakingRelu\\(0\\.010*\\)"));
}

TEST_F(LeakingReluTest, apply_check)
{
  array_t x = {1.0, 1e-6, 0.0, -4.21, 17.321};
  array_t y;

  _sut->apply(x, y);
  EXPECT_THAT(y, ElementsAre(
    DoubleNear(1.0, 1e-9), 
    DoubleNear(1e-6, 1e-9), 
    DoubleNear(0.0, 1e-9), 
    DoubleNear(-4.21 * _leakingRate, 1e-9), 
    DoubleNear(17.321, 1e-9)));

  _sut->apply(x);
  EXPECT_THAT(x, ContainerEq(y));
}

TEST_F(LeakingReluTest, derivativeCheck)
{
  array_t y = {10.0, 1.0, 0.0, -18.0, 7.3};
  array_t dx;
  
  _sut->deriverate(y, dx);

  EXPECT_THAT(dx, ElementsAre(
    DoubleNear(1.0, 1e-9), 
    DoubleNear(1.0, 1e-9), 
    DoubleNear(_leakingRate, 1e-9), 
    DoubleNear(_leakingRate, 1e-9), 
    DoubleNear(1.0, 1e-9)));
}

} // namespace ccml