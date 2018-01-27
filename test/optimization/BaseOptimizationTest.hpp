#ifndef CCML_BASE_OPTIMIZATION_TEST
#define CCML_BASE_OPTIMIZATION_TEST

#include <algorithm>
#include <cmath>
#include <memory>

#include <CcmlTest.hpp>

#include <Network.hpp>
#include <Sample.hpp>

namespace ccml {

template<typename Optimization>
class BaseOptimizationTest: public CcmlTest
{
protected:
  sample_list_t sinus(size_t count)
  {
    sample_list_t samples;
    auto gen = initializer::uniform(0,1);

    samples.reserve(count);
    for(size_t i=0; i<count; ++i)
    {
      auto x = gen();
      samples.emplace_back(Sample({x}, {std::sin(x)}));
    }  
    return samples;
  }

  void validate(const Network& network, const sample_list_t& samples, double epsilon)
  {
    std::for_each(samples.cbegin(), samples.cend(), [&network,epsilon](const Sample& s){
      auto prediction = network.output(s.input);
      for(size_t i=0; i<prediction.size(); ++i) 
      {
        EXPECT_NEAR(prediction[i], s.output[i], epsilon);
      }
    });
  }

protected:
  std::unique_ptr<Optimization> _sut;
};

} // namespace ccml

#endif // CCML_BASE_OPTIMIZATION_TEST