#include <cmath>

#include "Activation.hpp"

namespace ccml {

Activation& Activation::sigmoid() 
{
  static Activation aSigmoid(
    [](double x) {
      return 1 / (1 + std::exp(-x));
    },
    [](double y) {
      return y / (1 - y);
    }
  );
}

} // namespace ccml