#ifndef CCML_SAMPLE_HPP
#define CCML_SAMPLE_HPP

#include <Types.hpp>

namespace ccml {

struct Sample 
{
  Sample(const input_t& i, const output_t& o): input(i), output(o) {}

  const input_t input;
  const output_t output;
};

typedef std::vector<Sample> sample_list_t;

} // namespace ccml

#endif // CCML_SAMPLE_HPP