#ifndef CCML_UTILS_HPP
#define CCML_UTILS_HPP

#include <algorithm>

namespace ccml {

template<typename Iter, typename Fun>
void indexed_for_each(Iter begin, Iter end, Fun fun)
{
  size_t idx(0);
  std::for_each(begin, end, [&](auto& elem) {
    fun(elem, idx++);
  });
}

} // namespace ccml

#endif // CCML_UTILS_HPP