#ifndef CCML_TRANSFER_HPP
#define CCML_TRANSFER_HPP

#include <memory>
#include <ostream>

#include <Serializable.hpp>
#include <Types.hpp>

namespace ccml {

class Transfer
{
public:
  const std::string& name() const noexcept;

  virtual void apply(array_t& x) const;

  virtual void apply(const array_t& x, array_t& y) const;

  virtual void deriverate(const array_t& y, array_t& dx) const;

protected:
  Transfer(const std::string& name): _name(name) {}

  virtual ~Transfer() {}

  virtual value_t apply(value_t x) const;

  virtual value_t deriverate(value_t y) const;

protected:
  const std::string _name;
};

using transfer_ptr_t = std::shared_ptr<Transfer>;

} // namespace ccml}

#endif // CCML_TRANSFER_HPP