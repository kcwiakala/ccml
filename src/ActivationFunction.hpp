#ifndef CCML_ACTIVATION_FUNCTION_HPP
#define CCML_ACTIVATION_FUNCTION_HPP

#include <functional>

namespace ccml {

typedef std::function<double(double)> activation_function_t;
typedef std::function<double(double)> activation_derivative_t;

class ActivationFunction
{
  const std::string name;
  const activation_function_t fn;
  const activation_derivative_t df;
};

} // namespace ccml

#endif