#ifndef CCML_TYPES_HPP
#define CCML_TYPES_HPP

#include <vector>

namespace ccml {

typedef double value_t;

typedef std::vector<value_t> array_t;

typedef array_t tensor_1d_t;
typedef std::vector<tensor_1d_t> tensor_2d_t;
typedef std::vector<tensor_2d_t> tensor_3d_t;
typedef std::vector<tensor_3d_t> tensor_4d_t;

typedef array_t input_t;
typedef array_t output_t;

} // namespace ccml

#endif // CCML_TYPES_HPP