#ifndef CCML_TYPES_HPP
#define CCML_TYPES_HPP

#include <vector>

namespace ccml {

typedef double value_t;

template<typename T>
using vector_2d = std::vector<std::vector<T>>;

typedef std::vector<value_t> array_t;
typedef vector_2d<value_t> array_2d_t;

typedef array_t input_t;
typedef array_t output_t;

class MatrixView
{
public:
  MatrixView(array_t& data, size_t w, size_t h);

  const value_t& at(size_t x, size_t y) const
  {
    return _data.at(index(x,y));
  }

  value_t& at(size_t x, size_t y)
  {
    return _data.at(index(x,y));
  }

private:
  size_t index(size_t x, size_t y) const
  {
    return x + y * _w;
  }

private:
  const size_t _w;
  array_t& _data;
};

} // namespace ccml

#endif // CCML_TYPES_HPP