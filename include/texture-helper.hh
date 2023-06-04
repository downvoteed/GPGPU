#pragma once

#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>

namespace texture_helper {
using feature_vector = std::vector<uint8_t>;

/**
 * Calculate the LBP value of a pixel in a frame
 * @param frame The OpenCV frame
 * @param c The c coordinate of the pixel
 * @param r The r coordinate of the pixel
 * @return The LBP value of the pixel
 */
const uint8_t calculateLBP(const cv::Mat &frame, const int c, const int r) {
  uint8_t lbp = 0;
  uint8_t center = frame.at<uint8_t>(r, c);

  // Define the relative positions of the 8 neighbors
  const int neighbors[8][2] = {{-1, -1}, {0, -1}, {1, -1}, {-1, 0},
                               {1, 0},   {-1, 1}, {0, 1},  {1, 1}};

  // Compare the intensity value of the center pixel with its neighbors
  for (int i = 0; i < 8; i++) {
    const int nc = c + neighbors[i][0];
    const int nr = r + neighbors[i][1];

    // Check if the neighbor coordinates are within the image boundaries
    if (nc >= 0 && nc < frame.cols && nr >= 0 && nr < frame.rows) {
      // If the neighbor is less than the center, set the bit to 1
      const uint8_t neighbor = frame.at<uint8_t>(nr, nc);
      lbp |= (neighbor < center) << i;
    }
  }

  return lbp;
}

/**
 * Compare the LBP features of two frames
 * @param i The index of the LBP feature
 * @param f1 The LBP features of the first frame
 * @param f2 The LBP features of the second frame
 * @return The similarity between the two frames
 */
const double compare(const unsigned int i, const feature_vector &f1,
                     const feature_vector &f2) {
  // Calculate the number of identical bits
  const uint8_t p1 = std::popcount(f1[i]);
  const uint8_t p2 = std::popcount(f2[i]);
  const double identical = (double)std::max(p1, p2);

  // Calculate the total number of bits
  const double total = 8;

  // Calculate the similarity between the two LBP values as the ratio of
  // identical bits to total bits
  return identical / total;
}

} // namespace texture_helper
