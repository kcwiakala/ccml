#ifndef CCML_SERIALIZABLE_HPP
#define CCML_SERIALIZABLE_HPP

#include <memory>
#include <ostream>
#include <type_traits>

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

template<typename S>
typename std::enable_if<std::is_base_of<ccml::Serializable, S>::value,std::ostream&>::type 
  operator<<(std::ostream& stream, const std::shared_ptr<S>& ptr)
{
  ptr->toStream(stream);
  return stream;
}

template<typename S>
typename std::enable_if<std::is_base_of<ccml::Serializable, S>::value,std::ostream&>::type 
  operator<<(std::ostream& stream, const std::unique_ptr<S>& ptr)
{
  ptr->toStream(stream);
  return stream;
}

#endif // CCML_SERIALIZABLE_HPP