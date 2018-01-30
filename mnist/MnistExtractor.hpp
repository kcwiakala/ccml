#ifndef CCML_MNIST_EXTRACTOR_HPP
#define CCML_MNIST_EXTRACTOR_HPP

#include <cstdint>
#include <fstream>

#include <Sample.hpp>

namespace ccml {

class MnistExtractor 
{
public:
  MnistExtractor(const std::string& basePath, bool normalize = true);

  const sample_list_t& trainingSet() const;

  const sample_list_t& testSet() const;

private:
  using u8_list_t = std::vector<uint8_t>;
  using u8_matrix_t = std::vector<u8_list_t>;

  uint32_t load32Msb(std::ifstream& stream) const;

  void loadLabels(std::ifstream& stream, u8_list_t& labels) const;

  void loadImages(std::ifstream& stream, u8_matrix_t& images) const;

  void buildSampleList(const u8_matrix_t& images, const u8_list_t& labels, sample_list_t& samples);

  Sample buildSample(const u8_list_t& image, uint8_t label) const;

private:
  sample_list_t _trainingSet;
  sample_list_t _testSet;
};

} // namespace

#endif // CCML_MNIST_EXTRACTOR_HPP