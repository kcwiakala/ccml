#ifndef CCML_INITIALIZER_HPP
#define CCML_INITIALIZER_HPP

#include <functional>

#include <Types.hpp>

namespace ccml {

using initializer_t = std::function<value_t()>;

namespace initializer {

initializer_t constant(value_t value);

initializer_t uniform(value_t min, value_t max);

initializer_t normal(value_t mean, value_t sigma);

} // namespace initializer
} // namespace ccml

#endif 