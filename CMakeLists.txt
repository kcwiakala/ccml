cmake_minimum_required (VERSION 2.6)

project (ccml)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TEST_SOURCES "test/*.cpp")

set (CMAKE_CXX_FLAGS "-Wall -O2")

include_directories("src")

add_executable(ccml ${SOURCES} ${TEST_SOURCES})
target_link_libraries(ccml gtest gmock pthread)