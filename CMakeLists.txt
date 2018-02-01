cmake_minimum_required (VERSION 2.6)

project (ccml)

set(CMAKE_CXX_STANDARD 17)

include_directories("src")
include_directories("tp")

file(GLOB_RECURSE sources "src/*.cpp")
file(GLOB_RECURSE test_sources "test/*.cpp")

add_library(ccml STATIC ${sources})

target_compile_options(ccml PUBLIC -std=c++1y -Wall -Wfloat-conversion -O0)

find_package(GTest) 

if(GTEST_FOUND) 
  add_executable(utest ${test_sources})
  add_dependencies(utest ccml)
  target_compile_definitions(utest PUBLIC UNIT_TESTS)
  target_include_directories(utest PUBLIC "test" ${GTEST_INCLUDE_DIRS})
  target_link_libraries(utest PUBLIC ${GTEST_LIBRARIES} gmock pthread ccml)

  file(GLOB_RECURSE mnist_sources "mnist/*.cpp")
  add_executable(MNIST ${mnist_sources})
  add_dependencies(MNIST ccml)
  target_include_directories(MNIST PUBLIC "mnist" ${GTEST_INCLUDE_DIRS})
  target_link_libraries(MNIST PUBLIC ${GTEST_LIBRARIES} gmock pthread ccml)

endif()