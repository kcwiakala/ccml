#ifndef CCML_UTILS_HPP
#define CCML_UTILS_HPP

#include <algorithm>

#include <Types.hpp>

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

template<typename T, typename Fun>
void for_each(vector_2d<T>& cont, Fun fun)
{
  for(auto& row: cont)
  {
    for(T& obj: row)
    {
      fun(obj);
    }
  }
}

} // namespace ccml

#endif // CCML_UTILS_HPP