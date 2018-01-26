#ifndef CCML_SERIALIZABLE_HPP
#define CCML_SERIALIZABLE_HPP

#include <memory>
#include <ostream>
#include <type_traits>

#include <Types.hpp>

namespace ccml {

class Serializable 
{
public:
  virtual void toStream(std::ostream& stream) const = 0;
};

} // namespace ccml

inline std::ostream& operator<<(std::ostream& stream, const ccml::Serializable& obj)
{
  obj.toStream(stream);
  return stream;
}

template<typename T>
using enable_if_serializable = std::enable_if_t<std::is_base_of<ccml::Serializable, T>::value>;

template<typename T, typename = enable_if_serializable<T>>
std::ostream& operator<<(std::ostream& stream, const std::shared_ptr<T>& ptr)
{
  ptr->toStream(stream);
  return stream;
}

template<typename T, typename = enable_if_serializable<T>>
std::ostream& operator<<(std::ostream& stream, const std::unique_ptr<T>& ptr)
{
  ptr->toStream(stream);
  return stream;
}

inline std::ostream& operator<<(std::ostream& stream, const ccml::array_t& array)
{
  stream << "[";
  for(size_t i=0; i<array.size(); ++i)
  {
    stream << ((i>0) ? "," : "") << array[i];
  }
  stream << "]";
  return stream;
}

#endif // CCML_SERIALIZABLE_HPP