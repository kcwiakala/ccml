#include <memory>

#include <gtest/gtest.h>

#include <Initialization.hpp>
#include <Perceptron.hpp>

namespace ccml {
namespace {
const double delta=0.00001;
}

class PerceptronTest: public testing::Test
{
protected:
  PerceptronTest(): _sut(new Perceptron(2))
  {
    _sut->init(initializer::uniform(0.1, 0.9));
  }
protected:
  std::unique_ptr<Perceptron> _sut;
};

TEST_F(PerceptronTest, learn_lineary_separable) 
{
  sample_list_t and_set = {
    {{0,0}, {0}},
    {{1,0}, {0}},
    {{0,1}, {0}},
    {{1,1}, {1}}
  };

  EXPECT_TRUE(_sut->learn(and_set));
  EXPECT_NEAR(_sut->output({0,0}), 0.0, delta);
  EXPECT_NEAR(_sut->output({1,0}), 0.0, delta);
  EXPECT_NEAR(_sut->output({0,1}), 0.0, delta);
  EXPECT_NEAR(_sut->output({1,1}), 1.0, delta);
}

TEST_F(PerceptronTest, fail_learn_lineary_non_separable) 
{
  sample_list_t xor_set = {
    {{0,0}, {0}},
    {{1,0}, {1}},
    {{0,1}, {1}},
    {{1,1}, {0}}
  };
  EXPECT_FALSE(_sut->learn(xor_set));
}

}