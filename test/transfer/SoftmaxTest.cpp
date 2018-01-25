#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <numeric>

#include <transfer/Softmax.hpp>

namespace ccml {

using ::testing::ContainerEq;
using ::testing::ElementsAre;
using ::testing::DoubleNear;

class SoftmaxTest: public ::testing::Test
{
protected:
  SoftmaxTest(): _sut(std::make_shared<transfer::Softmax>())
  {
  }

  transfer_ptr_t _sut;
};

TEST_F(SoftmaxTest, name_check)
{
  EXPECT_EQ(_sut->name(), "softmax");
}

TEST_F(SoftmaxTest, apply_check)
{
  array_t x = {1.0, 2.0, -1.0, 0.0, 0.5};
  array_t y;

  _sut->apply(x, y);
  
  EXPECT_NEAR(std::accumulate(y.cbegin(), y.cend(), 0.0), 1.0, 1e-4);
  EXPECT_GT(y[1], y[0]);
  EXPECT_GT(y[0], y[4]);
  EXPECT_GT(y[4], y[3]);
  EXPECT_GT(y[3], y[2]);

  _sut->apply(x);
  EXPECT_THAT(x, ContainerEq(y));
}

TEST_F(SoftmaxTest, derivativeCheck)
{
  // TODO
}

} // namespace ccml