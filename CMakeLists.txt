cmake_minimum_required (VERSION 2.6)

project (ccml)

file(GLOB_RECURSE TEST_SOURCES "test/*.cpp")

add_executable(test.out ${TEST_SOURCES})
target_link_libraries(test.out gtest gmock pthread)