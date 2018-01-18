#include <iostream>

#include "NetworkOptimization.hpp"

namespace ccml {

void NetworkOptimization::advanceBatch(sample_batch_t& batch, const sample_list_t& samples, size_t batchSize) const
{
  if(batch.second == samples.end())
  {
    batch.first = batch.second = samples.begin();
  }
  const size_t remaining = std::distance(batch.second, samples.end());
  batch.first = batch.second;
  batch.second = (remaining < batchSize) ? samples.end() : std::next(batch.first, batchSize);
}

bool NetworkOptimization::train(const sample_list_t& samples, size_t batchSize, size_t maxEpochs, double epsilon)
{
  initTraining();

  bool success(false);

  if(batchSize > samples.size()) 
  {
    batchSize = samples.size();
  }

  sample_batch_t batch(samples.begin(), samples.end());

  for(size_t i=0; i<maxEpochs; ++i)
  {
    initEpoch();
    advanceBatch(batch, samples, batchSize);
    learnBatch(batch);

    const double totalLoss = _loss->compute(_network, samples);
    // std::cout << "Total loss " << i << " : " << totalLoss << std::endl;
    if(totalLoss < epsilon)
    {
      success = true;
      break;
    }
  }

  return success;
}

void NetworkOptimization::initTraining()
{
  // Nothing to do by default
}

void NetworkOptimization::initEpoch()
{
  // Nothing to do by default
}

} // namespace ccml