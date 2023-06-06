#pragma once

#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>

namespace texture_helper {
using feature_vector = std::vector<uint8_t>;

/**
 * Get the pixel value of a frame
 * @param frame The OpenCV frame
 * @param c The c coordinate of the pixel
 * @param r The r coordinate of the pixel
 * @return The pixel value
 */
uint8_t getPixel(const cv::Mat &frame, const int c, const int r) {
  try {
    return frame.at<uint8_t>(r, c);
  } catch (const std::exception &e) {
    return 0;
  }
}

/**
 * Calculate the LBP value of a pixel in a frame
 * @param frame The OpenCV frame
 * @param c The c coordinate of the pixel
 * @param r The r coordinate of the pixel
 * @return The LBP value of the pixel
 */
uint8_t calculateLBP(const cv::Mat &frame, const int c, const int r) {
  uint8_t lbp = 0;
  uint8_t center = frame.at<uint8_t>(r, c);

  lbp = (lbp << 1) | (getPixel(frame, r - 1, c - 1) < center);
  lbp = (lbp << 1) | (getPixel(frame, r - 1, c) < center);
  lbp = (lbp << 1) | (getPixel(frame, r - 1, c + 1) < center);
  lbp = (lbp << 1) | (getPixel(frame, r, c - 1) < center);
  lbp = (lbp << 1) | (getPixel(frame, r, c + 1) < center);
  lbp = (lbp << 1) | (getPixel(frame, r + 1, c - 1) < center);
  lbp = (lbp << 1) | (getPixel(frame, r + 1, c) < center);
  lbp = (lbp << 1) | (getPixel(frame, r + 1, c + 1) < center);

  return lbp;
}

/**
 * Compare the LBP features of two frames
 * @param i The index of the LBP feature
 * @param f1 The LBP features of the first frame
 * @param f2 The LBP features of the second frame
 * @return The similarity between the two frames
 */
float compare(const unsigned int i, const feature_vector &f1,
              const feature_vector &f2) {
  // Calculate the number of identical bits using biwise and popcount
  const uint8_t vector = ~(f1[i] ^ f2[i]);
  return __builtin_popcount(vector) / 8.0f;
}

} // namespace texture_helper
