#ifndef CCML_SAMPLE_HPP
#define CCML_SAMPLE_HPP

#include <Types.hpp>

namespace ccml {

struct Sample 
{
  Sample(const array_t& i, const array_t& o): input(i), output(o) {}

  Sample(array_t&& i, array_t&& o): input(std::move(i)), output(std::move(o)) {}

  const array_t input;
  const array_t output;
};

typedef std::vector<Sample> sample_list_t;
typedef std::pair<sample_list_t::const_iterator, sample_list_t::const_iterator> sample_batch_t;

} // namespace ccml

#endif // CCML_SAMPLE_HPP