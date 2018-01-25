/*
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <Transfer.hpp>

namespace ccml {

class HeavisideTest: public testing::TestWithParam<Transfer> {};

TEST_P(HeavisideTest, name_check)
{
  EXPECT_EQ(GetParam().name, "heaviside");
}

TEST_P(HeavisideTest, operation_check)
{
  auto h = GetParam();
  EXPECT_NEAR(h.operation(1.0), 1.0, 1e-9);
  EXPECT_NEAR(h.operation(1e-6), 1.0, 1e-9);
  EXPECT_NEAR(h.operation(0.0), 0.0, 1e-9);
  EXPECT_NEAR(h.operation(-1e-6), 0.0, 1e-9);
  EXPECT_NEAR(h.operation(-1.0), 0.0, 1e-9);
}

TEST_P(HeavisideTest, derivativeCheck)
{
  auto h = GetParam();

  EXPECT_NEAR(h.derivativeFromX(0.0), 0.0, 1e-9);
  EXPECT_NEAR(h.derivativeFromX(10.0), 0.0, 1e-9);
  EXPECT_NEAR(h.derivativeFromX(- 10.0), 0.0, 1e-9);

  EXPECT_NEAR(h.derivativeFromY(0.0), 0.0, 1e-9);
  EXPECT_NEAR(h.derivativeFromY(10.0), 0.0, 1e-9);
  EXPECT_NEAR(h.derivativeFromY(- 10.0), 0.0, 1e-9);
}

INSTANTIATE_TEST_CASE_P(TransferTest, HeavisideTest, 
  testing::Values(transfer::heaviside(), transfer::create("heaviside")));

class SigmoidTest: public testing::TestWithParam<Transfer> {};

TEST_P(SigmoidTest, name_check)
{
  EXPECT_EQ(GetParam().name, "sigmoid");
}

TEST_P(SigmoidTest, operation_check)
{
  auto s = GetParam();
  EXPECT_NEAR(s.operation(1.0), 0.7311, 1e-4);
  EXPECT_NEAR(s.operation(1e9), 1.0, 1e-4);
  EXPECT_NEAR(s.operation(0.0), 0.5, 1e-9);
  EXPECT_NEAR(s.operation(-1e9), 0.0, 1e-4);
  EXPECT_NEAR(s.operation(-1.0), 0.2689, 1e-4);
}

TEST_P(SigmoidTest, derivativeCheck)
{
  auto s = GetParam();

  EXPECT_NEAR(s.derivativeFromX(0.0), 0.25, 1e-9);
  EXPECT_NEAR(s.derivativeFromX(10.0), 0.0, 1e-4);
  EXPECT_NEAR(s.derivativeFromX(-10.0), 0.0, 1e-4);
  EXPECT_NEAR(s.derivativeFromX(1.0), 0.1966, 1e-4);
  EXPECT_NEAR(s.derivativeFromX(-1.0), 0.1966, 1e-4);

  EXPECT_NEAR(s.derivativeFromY(0.0), 0.0, 1e-9);
  EXPECT_NEAR(s.derivativeFromY(1.0), 0.0, 1e-9);
  EXPECT_NEAR(s.derivativeFromY(0.5), 0.25, 1e-9);
  EXPECT_NEAR(s.derivativeFromY(0.8), 0.16, 1e-9);
  EXPECT_NEAR(s.derivativeFromY(0.3), 0.21, 1e-9);
}

INSTANTIATE_TEST_CASE_P(TransferTest, SigmoidTest, 
  testing::Values(transfer::sigmoid(), transfer::create("sigmoid")));


class IdentityTest: public testing::TestWithParam<Transfer> {};

TEST_P(IdentityTest, name_check)
{
  EXPECT_EQ(GetParam().name, "identity");
}

TEST_P(IdentityTest, operation_check)
{
  auto i = GetParam();

  EXPECT_NEAR(i.operation(1.0), 1.0, 1e-9);
  EXPECT_NEAR(i.operation(1e9), 1e9, 1e-9);
  EXPECT_NEAR(i.operation(0.0), 0.0, 1e-9);
  EXPECT_NEAR(i.operation(-1e9), -1e9, 1e-9);
  EXPECT_NEAR(i.operation(-1.0), -1, 1e-9);
}

TEST_P(IdentityTest, derivativeCheck)
{
  auto i = GetParam();

  EXPECT_NEAR(i.derivativeFromX(0.0), 1.0, 1e-9);
  EXPECT_NEAR(i.derivativeFromX(10.0), 1.0, 1e-9);
  EXPECT_NEAR(i.derivativeFromX(-10.0), 1.0, 1e-9);
  EXPECT_NEAR(i.derivativeFromX(1.0), 1.0, 1e-9);
  EXPECT_NEAR(i.derivativeFromX(-1.0), 1.0, 1e-9);

  EXPECT_NEAR(i.derivativeFromY(0.0), 1.0, 1e-9);
  EXPECT_NEAR(i.derivativeFromY(1.0), 1.0, 1e-9);
  EXPECT_NEAR(i.derivativeFromY(0.5), 1.0, 1e-9);
  EXPECT_NEAR(i.derivativeFromY(-0.8), 1.0, 1e-9);
  EXPECT_NEAR(i.derivativeFromY(-0.3), 1.0, 1e-9);
}

INSTANTIATE_TEST_CASE_P(TransferTest, IdentityTest, 
  testing::Values(transfer::identity(), transfer::create("identity")));


class ReluTest: public testing::TestWithParam<Transfer> {};

TEST_P(ReluTest, name_check)
{
  EXPECT_EQ(GetParam().name, "relu");
}

TEST_P(ReluTest, operation_check)
{
  auto r = GetParam();

  EXPECT_NEAR(r.operation(1.0), 1.0, 1e-9);
  EXPECT_NEAR(r.operation(1e9), 1e9, 1e-9);
  EXPECT_NEAR(r.operation(0.0), 0.0, 1e-9);
  EXPECT_NEAR(r.operation(-1e9), 0.0, 1e-9);
  EXPECT_NEAR(r.operation(-1.0), 0.0, 1e-9);
}

TEST_P(ReluTest, derivativeCheck)
{
  auto r = GetParam();

  EXPECT_NEAR(r.derivativeFromX(0.0), 0.0, 1e-9);
  EXPECT_NEAR(r.derivativeFromX(10.0), 1.0, 1e-9);
  EXPECT_NEAR(r.derivativeFromX(-10.0), 0.0, 1e-9);
  EXPECT_NEAR(r.derivativeFromX(1.0), 1.0, 1e-9);
  EXPECT_NEAR(r.derivativeFromX(-1.0), 0.0, 1e-9);

  EXPECT_NEAR(r.derivativeFromY(0.0), 0.0, 1e-9);
  EXPECT_NEAR(r.derivativeFromY(1.0), 1.0, 1e-9);
  EXPECT_NEAR(r.derivativeFromY(0.5), 1.0, 1e-9);
  EXPECT_NEAR(r.derivativeFromY(-0.8), 0.0, 1e-9);
  EXPECT_NEAR(r.derivativeFromY(-0.3), 0.0, 1e-9);
}

INSTANTIATE_TEST_CASE_P(TransferTest, ReluTest, 
  testing::Values(transfer::relu(), transfer::create("relu")));


class LeakingReluTest: public testing::TestWithParam<Transfer> {};

double getLeakingRate(const std::string& name)
{
  const size_t parenthesisPos = name.find_first_of('(');
  const size_t parenthesisEnd = name.find_last_of(')');
  return std::stod(name.substr(parenthesisPos+1, parenthesisEnd - parenthesisPos - 1));
}

TEST_P(LeakingReluTest, name_check)
{
  EXPECT_THAT(GetParam().name, testing::ContainsRegex("leakingRelu\\([\\.0-9]+\\)"));
}

TEST_P(LeakingReluTest, operation_check)
{
  auto lr = GetParam();
  auto rate = getLeakingRate(lr.name);

  EXPECT_NEAR(lr.operation(1.0), 1.0, 1e-9);
  EXPECT_NEAR(lr.operation(1e9), 1e9, 1e-9);
  EXPECT_NEAR(lr.operation(0.0), 0.0, 1e-9);
  EXPECT_NEAR(lr.operation(-1e9), rate * (-1e9), 1e-9);
  EXPECT_NEAR(lr.operation(-1.0), -rate, 1e-9);
}

TEST_P(LeakingReluTest, derivativeCheck)
{
  auto lr = GetParam();
  auto rate = getLeakingRate(lr.name);
  
  EXPECT_NEAR(lr.derivativeFromX(0.0), rate, 1e-9);
  EXPECT_NEAR(lr.derivativeFromX(10.0), 1.0, 1e-9);
  EXPECT_NEAR(lr.derivativeFromX(-10.0), rate, 1e-9);
  EXPECT_NEAR(lr.derivativeFromX(1.0), 1.0, 1e-9);
  EXPECT_NEAR(lr.derivativeFromX(-1.0), rate, 1e-9);

  EXPECT_NEAR(lr.derivativeFromY(0.0), rate, 1e-9);
  EXPECT_NEAR(lr.derivativeFromY(1.0), 1.0, 1e-9);
  EXPECT_NEAR(lr.derivativeFromY(0.5), 1.0, 1e-9);
  EXPECT_NEAR(lr.derivativeFromY(-0.8), rate, 1e-9);
  EXPECT_NEAR(lr.derivativeFromY(-0.3), rate, 1e-9);
}

INSTANTIATE_TEST_CASE_P(TransferTest, LeakingReluTest, 
  testing::Values(transfer::leakingRelu(0.01), transfer::create("leakingRelu(0.1)")));

TEST(TransferTest, customTransfer)
{
  Transfer myTransfer("myTransfer", [](value_t x) { 
    return 17 * x;
  },
  [](value_t x) {
    return 5 * x;
  },
  [](value_t y) {
    return -3 * y;
  });

  EXPECT_TRUE(transfer::registerTransfer(myTransfer));
  
  auto t = transfer::create("myTransfer");

  EXPECT_EQ(t.name, "myTransfer");
  
  EXPECT_NEAR(t.operation(1.0), 17.0, 1e-9);
  EXPECT_NEAR(t.operation(-8.0), -136.0, 1e-9);
  EXPECT_NEAR(t.operation(0.0), 0.0, 1e-9);

  EXPECT_NEAR(t.derivativeFromX(1.0), 5.0, 1e-9);
  EXPECT_NEAR(t.derivativeFromX(-3.0), -15.0, 1e-9);
  EXPECT_NEAR(t.derivativeFromX(0.0), 0.0, 1e-9);

  EXPECT_NEAR(t.derivativeFromY(9.0), -27.0, 1e-9);
  EXPECT_NEAR(t.derivativeFromY(-1.0), 3.0, 1e-9);
  EXPECT_NEAR(t.derivativeFromY(0.0), 0.0, 1e-9);
}

} // namespace ccml
*/