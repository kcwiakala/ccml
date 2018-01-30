#include <arpa/inet.h>

#include "MnistExtractor.hpp"

namespace ccml {

MnistExtractor::MnistExtractor(const std::string& basePath, bool normalize)
{
  u8_matrix_t images;
  u8_list_t labels;

  std::ifstream trainLabels(basePath + "/train-labels-idx1-ubyte");
  std::ifstream trainImages(basePath + "/train-images-idx3-ubyte");
  loadLabels(trainLabels, labels);
  loadImages(trainImages, images);
  buildSampleList(images, labels, _trainingSet);

  std::ifstream testLabels(basePath + "/t10k-labels-idx1-ubyte");
  std::ifstream testImages(basePath + "/t10k-images-idx3-ubyte");
  loadLabels(testLabels, labels);
  loadImages(testImages, images);
  buildSampleList(images, labels, _testSet);
}

const sample_list_t& MnistExtractor::trainingSet() const
{
  return _trainingSet;
}

const sample_list_t& MnistExtractor::testSet() const
{
  return _testSet;
}

uint32_t MnistExtractor::load32Msb(std::ifstream& stream) const
{
  uint32_t aux;
  stream.read(reinterpret_cast<char*>(&aux), sizeof(aux));
  return ntohl(aux);
}

void MnistExtractor::loadLabels(std::ifstream& stream, u8_list_t& labels) const
{
  const uint32_t magicNumber = load32Msb(stream);
  if(magicNumber != 0)
  {
    const uint32_t labelCount = load32Msb(stream);
    labels.assign(labelCount, 0);
    for(uint8_t& label: labels)
    {
      stream.get(reinterpret_cast<char&>(label));
    }
  }
}

void MnistExtractor::loadImages(std::ifstream& stream, u8_matrix_t& images) const
{
  images.clear();

  const uint32_t magicNumber = load32Msb(stream);
  if(magicNumber != 0)
  {
    uint32_t imageCount = load32Msb(stream);
    images.reserve(imageCount);
    
    const uint32_t rowCount = load32Msb(stream);
    const uint32_t columnCount = load32Msb(stream);
    const uint32_t pixelCount = rowCount * columnCount;
    while(imageCount--)
    {
      images.emplace_back(u8_list_t(pixelCount, 0));
      for(uint8_t& pixel: images.back())
      {
        stream.get(reinterpret_cast<char&>(pixel));
      }
    }
  }
}

void MnistExtractor::buildSampleList(const u8_matrix_t& images, const u8_list_t& labels, sample_list_t& samples)
{
  samples.clear();
  samples.reserve(images.size());
  for(size_t i=0; i<images.size(); ++i)
  {
    samples.emplace_back(buildSample(images[i], labels[i]));
  }
}

Sample MnistExtractor::buildSample(const u8_list_t& image, uint8_t label) const
{
  array_t output(10, 0.0);
  output[label] = 1.0;
  return Sample(array_t(image.begin(), image.end()), std::move(output));
}

} // namespace ccml