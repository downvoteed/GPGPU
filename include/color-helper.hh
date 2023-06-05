#pragma once

#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>

namespace color_helper {

using color_components = std::vector<uint8_t>;
using similarity_vector = std::vector<double>;
using similarity_vectors = std::vector<similarity_vector>;

/**
 * Extract the color components (R/G) from a frame at position c, r
 * @param frame The OpenCV frame
 * @param c The c coordinate
 * @param r The r coordinate
 * @return The color components
 */
const color_components *convert(const cv::Mat &frame, const int c,
                                const int r) {
  color_components *components = new color_components();

  // Extract the color components from the frame for the given coordinates
  cv::Vec3b pixel = frame.at<cv::Vec3b>(r, c);
  uint8_t r_component = pixel[2];
  uint8_t g_component = pixel[1];

  components->push_back(r_component);
  components->push_back(g_component);
  return components;
}

/**
 * Compare the color components of two frames at position c, r
 * @param frame1 The first OpenCV frame
 * @param frame2 The second OpenCV frame
 * @param c The c coordinate
 * @param r The r coordinate
 * @return The color similarities
 */
const similarity_vector *compare(const cv::Mat &frame1, const cv::Mat &frame2,
                                 const int c, const int r) {
  similarity_vector *similarities = new similarity_vector();

  // Calculate the color components for the two frames at the given coordinates
  const color_components *c1 = convert(frame1, c, r);
  const color_components *c2 = convert(frame2, c, r);

  // Calculate the color similarities for each color component
  uint8_t r1 = c1->at(0);
  uint8_t r2 = c2->at(1);
  uint8_t r_max = std::max(r1, r2);
  uint8_t r_min = std::min(r1, r2);
  similarities->push_back((double)r_min / (double)r_max);

  uint8_t g1 = c1->at(0);
  uint8_t g2 = c2->at(1);
  uint8_t g_max = std::max(g1, g2);
  uint8_t g_min = std::min(g1, g2);
  similarities->push_back((double)g_min / (double)g_max);

  // Free the memory
  delete c1;
  delete c2;

  return similarities;
}

} // namespace color_helper
