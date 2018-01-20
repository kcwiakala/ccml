#ifndef CCML_TYPES_HPP
#define CCML_TYPES_HPP

#include <functional>
#include <vector>
#include <ostream>

namespace ccml {

using value_t = double;

template<typename T>
using vector_2d = std::vector<std::vector<T>>;

using array_t = std::vector<value_t>;
using array_2d_t = vector_2d<value_t>;

using value_converter_t = std::function<value_t(value_t)>;
using value_operation_t = std::function<value_t(value_t, value_t)>;

} // namespace ccml

#endif // CCML_TYPES_HPP