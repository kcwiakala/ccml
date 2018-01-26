#ifndef CCML_CCML_TEST_HPP
#define CCML_CCML_TEST_HPP

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#define EXPECT_THROW_MATCHING(statement, exc, matcher) \
  EXPECT_THROW({ \
    try { \
    statement; \
    } catch(exc& e) { \
      EXPECT_THAT(e.what(), matcher); \
      throw; \
    }}, exc)

namespace ccml {

class CcmlTest: public ::testing::Test
{

};

} // namespace ccml

#endif // CCML_CCML_TEST_HPP