#include <iostream>
#include <memory>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <layer/NeuronLayer.hpp>

#include <transfer/Heaviside.hpp>
#include <transfer/Sigmoid.hpp>

namespace ccml {

class NeuronLayerMock: public NeuronLayer
{
public:
  NeuronLayerMock(size_t layerSize, size_t neuronSize, const transfer_ptr_t& transfer):
    NeuronLayer("NeuronLayerMock", layerSize, neuronSize, transfer)
  {
  }

  MOCK_CONST_METHOD0(inputSize, size_t());
  MOCK_CONST_METHOD2(output, void(const array_t&, array_t&));
  MOCK_CONST_METHOD2(backpropagate, void(const array_t&, array_t&));
  MOCK_CONST_METHOD3(splitError, void(const array_t&, const array_t&, const error_reader_t&));
};

class NeuronLayerTest: public testing::Test
{
protected:
  NeuronLayerTest():
    _layer(new NeuronLayerMock(3,2,std::make_shared<transfer::Sigmoid>()))
  {
  }

protected:
  std::unique_ptr<NeuronLayerMock> _layer;
};

TEST_F(NeuronLayerTest, number_of_neurons)
{
  _layer.reset(new NeuronLayerMock(6, 2, std::make_shared<transfer::Heaviside>()));

  EXPECT_EQ(_layer->outputSize(), 6u);
  EXPECT_EQ(_layer->size(), 6u);
}

TEST_F(NeuronLayerTest, neuron_initialization)
{
  _layer.reset(new NeuronLayerMock(3,2,std::make_shared<transfer::Heaviside>()));

  _layer->init(initializer::constant(87), initializer::constant(5));

  for(size_t i=0; i<_layer->size(); ++i)
  {
    const auto& node = _layer->node(i);
    EXPECT_NEAR(node.bias(), 5.0, 1e-6);
    std::for_each(node.weights().begin(), node.weights().end(), [](double w) {
      EXPECT_NEAR(w, 87.0, 1e-6);
    });
  }
}

} // namespace ccml