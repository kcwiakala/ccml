#include <gtest/gtest.h>
#include <iostream>

#include <boost/bind.hpp>

int main(int argc, char **argv) 
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}