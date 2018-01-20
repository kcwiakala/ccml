#include <iostream>
#include <memory>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <layer/NeuronLayer.hpp>

namespace ccml {

class NeuronLayerMock: public NeuronLayer
{
public:
  NeuronLayerMock(size_t layerSize, size_t neuronSize, const Transfer& transfer):
    NeuronLayer("NeuronLayerMock", layerSize, neuronSize, transfer)
  {
  }

  MOCK_CONST_METHOD0(inputSize, size_t());
  MOCK_CONST_METHOD2(output, void(const array_t&, array_t&));
  MOCK_CONST_METHOD2(backpropagate, void(const array_t&, array_t&));
};

class NeuronLayerTest: public testing::Test
{
protected:
  NeuronLayerTest():
    _layer(new NeuronLayerMock(3,2,transfer::sigmoid()))
  {
  }

protected:
  std::unique_ptr<NeuronLayerMock> _layer;
};

TEST_F(NeuronLayerTest, number_of_neurons)
{
  _layer.reset(new NeuronLayerMock(6, 2, transfer::heaviside()));

  EXPECT_EQ(_layer->outputSize(), 6u);
  EXPECT_EQ(_layer->size(), 6u);
}

TEST_F(NeuronLayerTest, neuron_initialization)
{
  _layer.reset(new NeuronLayerMock(3,2,transfer::heaviside()));

  _layer->init(initializer::constant(87), initializer::constant(5));

  for(size_t i=0; i<_layer->size(); ++i)
  {
    const auto& neuron = _layer->neuron(i);
    EXPECT_NEAR(neuron.bias(), 5.0, 1e-6);
    std::for_each(neuron.weights().begin(), neuron.weights().end(), [](double w) {
      EXPECT_NEAR(w, 87.0, 1e-6);
    });
  }
}

} // namespace ccml