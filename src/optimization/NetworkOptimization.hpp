#ifndef CCML_NETWORK_OPTIMIZATION_HPP
#define CCML_NETWORK_OPTIMIZATION_HPP

#include <loss/AbstractLoss.hpp>
#include <Network.hpp>
#include <Sample.hpp>

namespace ccml {

class NetworkOptimization 
{
public:
  bool train(const sample_list_t& samples, size_t batchSize, size_t maxIterations, double epsilon);

protected:
  virtual void learnBatch(const sample_batch_t& batch) = 0;

  virtual void initTraining();

  virtual void initEpoch();

protected:
  NetworkOptimization(Network& network, loss_ptr_t loss);

  virtual ~NetworkOptimization() {}

private:
  void advanceBatch(sample_batch_t& batch, const sample_list_t& samples, size_t batchSize) const;

protected:
  Network& _network;
  const loss_ptr_t _loss;
  const bool _transferIncluded;
};

} // namespace ccml

#endif // CCML_NETWORK_OPTIMIZATION_HPP