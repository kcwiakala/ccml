#ifndef CCML_UTILS_HPP
#define CCML_UTILS_HPP

#include <algorithm>

namespace ccml {

template<typename Iter, typename Fun>
void indexed_for_each(Iter first, Iter last, Fun fun)
{
  size_t idx(0);
  while(first != last)
  {
    fun(*first++, idx++);
  }
}

template<typename Iter, typename T>
void multiply_each(Iter first, Iter last, T val)
{
  while(first != last) 
  {
    *first = *first++ * val;
  }
}

} // namespace ccml

#endif // CCML_UTILS_HPP