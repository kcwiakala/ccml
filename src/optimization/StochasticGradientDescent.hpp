#ifndef CCML_STOCHASTIC_GRADIENT_DESCENT_HPP
#define CCML_STOCHASTIC_GRADIENT_DESCENT_HPP

#include "Backpropagation.hpp"

namespace ccml {

struct GradientData
{
  GradientData(const Node& node);

  void reset();

  array_t weights;
  value_t bias;
};

class StochasticGradientDescent: public Backpropagation
{
public:
  StochasticGradientDescent(Network& network, const loss_ptr_t& loss, double rate);

  virtual ~StochasticGradientDescent();

protected:
  virtual void adjustNode(Node& node, GradientData& gradients, size_t layerIdx, size_t nodeIdx);

protected:
  virtual void learnBatch(const sample_batch_t& batch);

  void learnSample(const Sample& sample);

  void updateGradients(const array_t& input, const array_2d_t& activation, const array_2d_t& error);

  void normalizeGradients(size_t batchSize);

  void adjustNodes();

protected:
  const double _rate;
  vector_2d<GradientData> _gradients;
};

} // namespace ccml

#endif // CCML_STOCHASTIC_GRADIENT_DESCENT_HPP