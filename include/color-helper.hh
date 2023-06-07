#pragma once

#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>

namespace color_helper {

using similarity_vector = std::vector<float>;
using similarity_vectors = std::vector<similarity_vector>;

/**
 * Extract the color components (R/G) from a frame at position c, r
 * @param frame The OpenCV frame
 * @param c The c coordinate
 * @param r The r coordinate
 * @param r_component The R component
 * @param g_component The G component
 */
void convert(const cv::Mat &frame, const int c, const int r,
             uint8_t &r_component, uint8_t &g_component) {
  // Extract the color components from the frame for the given coordinates

  cv::Vec3b pixel = frame.at<cv::Vec3b>(r, c);
  r_component = pixel[2];
  g_component = pixel[1];
}

/**
 * Compare the color components of two frames at position c, r
 * @param frame1 The first OpenCV frame
 * @param frame2 The second OpenCV frame
 * @param c The c coordinate
 * @param r The r coordinate
 * @param r_ratio The R color similarity
 * @param g_ratio The G color similarity
 */
void compare(const cv::Mat &frame1, const cv::Mat &frame2, const int c,
             const int r, float &r_ratio, float &g_ratio) {
  uint8_t r1 = 0;
  uint8_t r2 = 0;
  uint8_t g1 = 0;
  uint8_t g2 = 0;

  // Calculate the color components for the two frames at the given coordinates
  convert(frame1, c, r, r1, g1);
  convert(frame2, c, r, r2, g2);

  // Calculate the color similarities for each color component min/max
  if (r1 > r2) {
    r_ratio = (float)r2 / (float)r1;
  } else {
    r_ratio = (float)r1 / (float)r2;
  }

  if (g1 > g2) {
    g_ratio = (float)g2 / (float)g1;
  } else {
    g_ratio = (float)g1 / (float)g2;
  }
}
} // namespace color_helper