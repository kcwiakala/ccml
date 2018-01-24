#include <iostream>
#include <memory>

#include <gtest/gtest.h>

#include <layer/FullyConnectedLayer.hpp>

#include <transfer/Sigmoid.hpp>
#include <transfer/Relu.hpp>

namespace ccml {

class FullyConnectedLayerTest: public testing::Test
{
protected:
  FullyConnectedLayerTest():
    _layer(new FullyConnectedLayer(3,2,std::make_shared<transfer::Sigmoid>()))
  {
  }

protected:
  std::unique_ptr<FullyConnectedLayer> _layer;
};

TEST_F(FullyConnectedLayerTest, input_size)
{
  _layer.reset(new FullyConnectedLayer(18,4,std::make_shared<transfer::Sigmoid>()));

  EXPECT_EQ(_layer->inputSize(), 18);

  for(size_t i=0; i<_layer->size(); ++i)
  {
    EXPECT_EQ(_layer->node(i).size(), 18);
  }
}

TEST_F(FullyConnectedLayerTest, activation)
{
  _layer.reset(new FullyConnectedLayer(3, 2, std::make_shared<transfer::Relu>()));
  _layer->init(initializer::constant(0.0), initializer::constant(0.0));

  ASSERT_EQ(_layer->size(), 2u);
  _layer->node(0).adjust({1,2,3}, 17);
  _layer->node(1).adjust({4,5,6}, 33);

  array_t x = {3,6,2};
  array_t y;
  _layer->output(x,y);

  ASSERT_EQ(y.size(), 2);
  EXPECT_NEAR(y[0], 1 * x[0] + 2 * x[1] + 3 * x[2] + 17, 1e-6);
  EXPECT_NEAR(y[1], 4 * x[0] + 5 * x[1] + 6 * x[2] + 33, 1e-6);

  // std::cout << _layer << std::endl;
}

} // namespace ccml