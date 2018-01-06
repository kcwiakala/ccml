#include <numeric>

#include <gtest/gtest.h>

#include <Initialization.hpp>

TEST(InitializationTest, constant_initializer)
{
  ccml::Initializer gen(ccml::Initializer::constant(8));

  for(int i=0; i < 100; ++i) 
  {
    EXPECT_NEAR(8.0, gen(), 0.0000001);
  }

  gen = ccml::Initializer::constant(23);
  EXPECT_NEAR(23.0, gen(), 0.0000001);

  gen = ccml::Initializer::constant(4837.234);
  EXPECT_NEAR(4837.234, gen(), 0.0000001);
}

TEST(InitializationTest, uniform_initializer)
{
  const double min = 234;
  const double max = 298;
  const double num = 10000;

  ccml::Initializer gen = ccml::Initializer::uniform(min, max);
  std::vector<double> results;
  results.reserve(num);

  for(int i=0; i<num; ++i) 
  {
    results.push_back(gen());
    EXPECT_GE(results.back(), min);
    EXPECT_LE(results.back(), max);
  }

  double sum = std::accumulate(results.begin(), results.end(), 0.0);
  EXPECT_NEAR((sum / num), min + (max - min) / 2, (max - min) / 100);  
}

TEST(InitializationTest, normal_initializer)
{
  const double mean = 23;
  const double sigma = 81;
  const double num = 10000;

  ccml::Initializer gen = ccml::Initializer::normal(mean, sigma);
  std::vector<double> results;
  results.reserve(num);

  for(int i=0; i<num; ++i) 
  {
    results.push_back(gen());
  }  

  double sum = std::accumulate(results.begin(), results.end(), 0.0);
  EXPECT_NEAR((sum / num), mean, sigma / 10);  
}
